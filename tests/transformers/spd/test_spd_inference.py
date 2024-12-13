# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import List, Optional
from pprint import pprint

import numpy as np
import pytest
from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM as AutoModelForCausalLM
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils.device_utils import get_available_device_id

configs = [
    pytest.param(
        #["My name is", "Hello", "Hi", "My name is"], # prompt
        ["My name is"], # prompt
        1, # num_speculative_tokens
        32, # prefill_seq_len
        128, # ctx_len
        1, # prefill_bsz
        "JackFram/llama-68m", # draft_model_name
        "JackFram/llama-68m", # target_model_name
        1, # full_batch_size
        id="CB llama",
    ),
]



def run_prefill_on_draft_and_target(
    tlm_session: QAICInferenceSession, 
    dlm_session: QAICInferenceSession, 
    inputs: dict, 
    prefill_seq_len: int, 
    slot_idx: int
):
    tlm_decode_start_input = dict()
    dlm_decode_start_input = dict()
    input_len = inputs.input_ids.shape[1]
    num_chunks = input_len // prefill_seq_len
    cache_index = np.array([[0]], np.int64)
    batch_index = np.array([[slot_idx]], np.int64)
    inputs["batch_index"] = batch_index

    # Run chunked prefill
    for i in range(num_chunks):
        chunk_inputs = inputs.copy()
        chunk_inputs["input_ids"] = inputs["input_ids"][:, cache_index[0, 0] : cache_index[0, 0] + prefill_seq_len]
        chunk_inputs["position_ids"] = inputs["position_ids"][:, cache_index[0, 0] : cache_index[0, 0] + prefill_seq_len]

        tlm_outputs = tlm_session.run(chunk_inputs)
        _ = dlm_session.run(chunk_inputs)
        cache_index += prefill_seq_len

    tlm_logits = tlm_outputs["logits"]
    return tlm_logits


def get_padded_input_len(input_len: int, prefill_seq_len: int, ctx_len: int):
    """return padded input length (must be factor of `prefill_seq_len`)

    Args:
        input_len (int): prompt length
        prefill_seq_len (int): prefill sequence length 
        ctx_len (int): context length

    Returns:
        input_len_padded (int): padded input length
    """
    num_chunks = -(input_len // -prefill_seq_len)  # ceil divide without float
    input_len_padded = num_chunks * prefill_seq_len  # Convert input_len to a multiple of prefill_seq_len
    assert input_len_padded <= ctx_len, "input_len rounded to nearest prefill_seq_len multiple should be less than ctx_len"
    return input_len_padded



def split_dlm_bonus_token_inputs(dlm_decode_inputs):
    bonus_token_inputs = dict()
    bonus, regular = np.hsplit(dlm_decode_inputs["input_ids"], 2)
    bonus_token_inputs["input_ids"] = bonus
    dlm_decode_inputs["input_ids"] = regular
    bonus, regular = np.hsplit(dlm_decode_inputs["position_ids"], 2)
    bonus_token_inputs["position_ids"] = bonus
    dlm_decode_inputs["position_ids"] = regular
    return bonus_token_inputs, dlm_decode_inputs

@pytest.mark.parametrize(
    "prompt, num_speculative_tokens, prefill_seq_len, ctx_len, prefill_bsz, draft_model_name, target_model_name, full_batch_size",
    configs,
)
def test_spec_decode_inference(
    prompt: List[str],
    num_speculative_tokens: int,
    prefill_seq_len: int,
    ctx_len: int,
    prefill_bsz: int,
    draft_model_name: str,
    target_model_name: str,
    full_batch_size: Optional[int],
):
    # get device group
    #device_group: List[int] = get_available_device_id()
    device_group: List[int] = [1]
    if not device_group:
        pytest.skip("No available devices to run model on Cloud AI 100")
    # assumes dlm and tlm are compiled to the same prompt-chunk-size, context length and full_batch_size/batch-size
    # get vocab size
    tokenizer = AutoTokenizer.from_pretrained(target_model_name, padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    vocab_size = len(tokenizer)

    # export_and_compile tlm and dlm
    continuous_batching = full_batch_size is not None
    target_model = AutoModelForCausalLM.from_pretrained(target_model_name, continuous_batching=continuous_batching, is_tlm=True)
    draft_model = AutoModelForCausalLM.from_pretrained(draft_model_name, continuous_batching=continuous_batching)

    num_devices = len(device_group)
    target_model_qpc_path: str = target_model.compile(num_cores=11,
                                                      num_devices=num_devices,
                                                      prefill_seq_len=prefill_seq_len,
                                                      ctx_len=ctx_len,
                                                      aic_enable_depth_first=True, 
                                                      full_batch_size=full_batch_size, 
                                                      num_speculative_tokens=num_speculative_tokens)
    draft_model_qpc_path: str = draft_model.compile(num_cores=5,
                                                    prefill_seq_len=prefill_seq_len,
                                                    ctx_len=ctx_len,
                                                    aic_enable_depth_first=True,
                                                    full_batch_size=full_batch_size)
    # init qaic session
    target_model_session = QAICInferenceSession(target_model_qpc_path, device_ids=device_group)
    draft_model_session = QAICInferenceSession(draft_model_qpc_path, device_ids=device_group)

    # skip inputs/outputs buffers
    target_model_session.skip_buffers(set([x for x in target_model_session.input_names if x.startswith("past_")]))
    target_model_session.skip_buffers(
        set([x for x in target_model_session.output_names if x.endswith("_RetainedState")])
    )
    draft_model_session.skip_buffers(set([x for x in draft_model_session.input_names if x.startswith("past_")]))
    draft_model_session.skip_buffers(set([x for x in draft_model_session.output_names if x.endswith("_RetainedState")]))

    is_cb = full_batch_size is not None
    if not is_cb:
        prompts = prompt * prefill_bsz
        decode_batch_size = prefill_bsz
    else:
        prompts = prompt
        decode_batch_size = full_batch_size
    # tokenize the prompts
    prompts_tokenized: List[dict] = []
    for p in prompts:
        input_len: int = tokenizer(p, return_tensors="np", padding=True).input_ids.shape[1]
        input_len_padded: int = get_padded_input_len(input_len, prefill_seq_len, ctx_len)
        p_tok: dict = tokenizer(p, return_tensors="np", padding="max_length", max_length=input_len_padded)
        position_ids = np.where(p_tok.pop("attention_mask"), np.arange(input_len_padded), -1)
        p_tok["position_ids"] = position_ids
        prompts_tokenized.append(p_tok)
    # create caches to hold generated ids and input prompt lengths
    generated_ids = [[] for i in range(decode_batch_size)]
    input_lengths = [0] * decode_batch_size
    # run prefill on both draft and target models
    dlm_decode_inputs = dict()
    dlm_decode_inputs["position_ids"] = np.zeros((decode_batch_size, 1), np.int64)
    dlm_decode_inputs["input_ids"] = np.full((decode_batch_size, 1), tokenizer.pad_token_id)
    dlm_decode_inputs["batch_index"] = np.reshape(
        np.array(np.arange(decode_batch_size), np.int64), (decode_batch_size, 1)
    )
    # mock input key "logits" to store the first batch of output logits
    tlm_precode_inputs = dict(
        input_ids = np.zeros((decode_batch_size, num_speculative_tokens+1), dtype=np.int64),
        position_ids = np.zeros((decode_batch_size, num_speculative_tokens+1), dtype=np.int64),
        batch_index = np.arange(decode_batch_size, dtype=np.int64).reshape(-1, 1)
    )
    generation_done = False
    max_gen_len = [ctx_len] * decode_batch_size
    num_logits_to_keep = num_speculative_tokens+1
    # setup buffers
    all_accept = np.full((decode_batch_size, num_speculative_tokens), False, dtype=bool)
    tlm_prefill_logits_ph = np.zeros((prefill_bsz, 1, vocab_size), dtype=np.float32)
    dlm_prefill_logits_ph = np.zeros((prefill_bsz, 1, vocab_size), dtype=np.float32)
    decode_logits_ph = np.zeros((decode_batch_size, 1, vocab_size), dtype=np.float32)
    precode_logits_ph = np.zeros((decode_batch_size, num_logits_to_keep, vocab_size), dtype=np.float32)

    target_model_session.set_buffers({"logits": tlm_prefill_logits_ph})
    draft_model_session.set_buffers({"logits": dlm_prefill_logits_ph})
    for bi in range(decode_batch_size):
        # assumes that prefill queue will always be popped from the front
        tlm_logits = run_prefill_on_draft_and_target(
            tlm_session=target_model_session,
            dlm_session=draft_model_session,
            inputs=prompts_tokenized[bi],
            prefill_seq_len=prefill_seq_len,
            slot_idx=bi,
        )
        input_ids = tlm_logits.argmax(2).astype(np.int64)
        dlm_decode_inputs["input_ids"] = input_ids
        tlm_precode_inputs["input_ids"][bi, 0] = input_ids.item()
        input_len = prompts_tokenized[bi]['position_ids'].max(1).item() + 1
        dlm_decode_inputs["position_ids"][bi, 0] = input_len
        tlm_precode_inputs["position_ids"][bi] = np.arange(input_len, input_len+num_speculative_tokens+1, dtype=np.int64)
        # assumes that prefill queue will always be popped from the front
        input_lengths[bi] = input_len
        max_gen_len[bi] -= input_lengths[bi]

    # set decode logits buffers
    target_model_session.set_buffers({"logits": precode_logits_ph})
    draft_model_session.set_buffers({"logits": decode_logits_ph})
    # start decode phase
    valid_batch_indices = list(range(decode_batch_size))[::-1]
    num_tokens_selected_per_validation = []
    all_accept = False
    it = 0
    break_idx = 1000
    while True:
        print('-'*60)
        print(f"{it=}")
        # generate proposals from draft model
        for k_ in range(num_speculative_tokens):
            if all_accept:
                #running decode one extra time in the first speculative iteration
                # workaround to avoid the incorrect precode with 3-specialized multi-batch DLM
                bonus_token_inputs, dlm_decode_inputs = split_dlm_bonus_token_inputs(dlm_decode_inputs)
                pprint(f"{bonus_token_inputs=}")
                _ = draft_model_session.run(bonus_token_inputs)
            dlm_outputs = draft_model_session.run(dlm_decode_inputs)
            pprint(f"{dlm_decode_inputs=}")
            pprint(f"{tlm_precode_inputs=}")
            input_ids = dlm_outputs["logits"].argmax(2)
            tlm_precode_inputs["input_ids"][:, k_+1] = input_ids.flatten()
            dlm_decode_inputs["input_ids"] = input_ids
            dlm_decode_inputs["position_ids"][valid_batch_indices] += 1
        # run precode on TLM to score the proposed tokens
        pprint(f"{dlm_decode_inputs=}")
        pprint(f"{tlm_precode_inputs=}")
        if it == break_idx: breakpoint()
        tlm_outputs = target_model_session.run(tlm_precode_inputs)
        target_logits = tlm_outputs["logits"][valid_batch_indices]
        # greedy sampling from target model
        target_tokens = target_logits.argmax(-1) # shape: [len(valid_batch_indices), num_speculative_tokens+1]
        # exact matching between draft and target tokens
        draft_tokens = tlm_precode_inputs["input_ids"][valid_batch_indices,1:] # shape: [len(valid_batch_indices), num_speculative_tokens]
        matching = draft_tokens == target_tokens[:, :-1]
        num_tokens_selected = matching.cumprod(axis=1).sum(axis=1) # shape: [len(valid_batch_indices)]
        all_accept = (num_tokens_selected == num_speculative_tokens).all()
        num_tokens_selected_per_validation.append(num_tokens_selected)
        print(f"{target_tokens=}")
        print(f"{num_tokens_selected=}")
        print(f"{all_accept=}")
        if it == break_idx: breakpoint()
        if num_tokens_selected.item() == 0: breakpoint()

        # append selected tokens to the generated_ids
        for bi in valid_batch_indices:
            accepted_tokens = num_tokens_selected[bi]
            if all_accept:
                accepted_tokens += 1
            num_tokens_to_append = min(accepted_tokens, max_gen_len[bi] - len(generated_ids[bi]))
            generated_ids[bi].extend(target_tokens[bi, :num_tokens_to_append].tolist())
            if len(generated_ids[bi]) >= max_gen_len[bi]:
                del valid_batch_indices[bi]
                continue
        # check if all generations are done
        if not valid_batch_indices: break
        # prepare decode inputs for next decode iteration
        common_input_ids = target_tokens[:, num_tokens_selected[valid_batch_indices]].reshape(len(valid_batch_indices), 1)
        common_position_ids = tlm_precode_inputs["position_ids"][valid_batch_indices, num_tokens_selected[valid_batch_indices]].reshape(len(valid_batch_indices), 1)+1
        if all_accept:
            # all_accept input_ids
            input_ids = np.zeros((decode_batch_size, 2), dtype=np.int64)
            input_ids[valid_batch_indices] = np.concatenate([target_tokens[:, num_tokens_selected-1].reshape(-1,1), common_input_ids], axis=1)
            dlm_decode_inputs["input_ids"] = input_ids
            # all_accept position_ids
            position_ids = np.zeros((decode_batch_size, 2), dtype=np.int64)
            position_ids[valid_batch_indices] = np.concatenate([tlm_precode_inputs["position_ids"][valid_batch_indices, num_tokens_selected-1].reshape(-1,1)+1, common_position_ids], axis=1)
            dlm_decode_inputs["position_ids"] = position_ids
        else:
            dlm_decode_inputs["input_ids"][valid_batch_indices] = common_input_ids
            dlm_decode_inputs["position_ids"][valid_batch_indices] = common_position_ids
        tlm_precode_inputs["input_ids"][valid_batch_indices,0] = common_input_ids
        tlm_precode_inputs["position_ids"][valid_batch_indices] += num_tokens_selected.reshape(len(valid_batch_indices),1)+1
        pprint(dlm_decode_inputs)
        pprint(tlm_precode_inputs)
        if it == break_idx: breakpoint()
        it += 1
    num_tokens_selected_per_validation = np.concatenate(num_tokens_selected_per_validation).reshape(len(num_tokens_selected_per_validation), decode_batch_size)
    mean_num_accepted_tokens_per_batch = num_tokens_selected_per_validation.mean(axis=0)
    print("mean number of accepted tokens per batch = ", mean_num_accepted_tokens_per_batch)
    print("max generation len = ", max_gen_len)
    print("actual generation len = ", [len(gid) for gid in generated_ids])
    print(tokenizer.batch_decode(generated_ids))