import os
import re
import sys
import time
import json
import pickle
from typing import Tuple
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import random
import numpy as np
import torch
import torch.distributed
import torch.backends.cudnn
import fire
from rap.models import QueryLlama
from rap.utils.gsm8k import judge_answer_gsm8k, get_gsm8k_dataset
# from rap.gsm8k_mcts import reasoning_mcts_search
from llama import ModelArgs, Transformer, Tokenizer, LLaMA
from fairscale.nn.model_parallel.initialize import initialize_model_parallel


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    return local_rank, world_size


def load(ckpt_dir: str, tokenizer_path: str, local_rank: int, world_size: int, max_batch_size: int) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    # print(checkpoints)
    assert (
            world_size == len(checkpoints)
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(max_seq_len=2048, max_batch_size=max_batch_size, **params)
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args).cuda().half()
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)
    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def main_iters(llama_ckpt='llama-ckpts/30B',
               prompts='data/gsm8k/prompts/interactive_examples.json',
               question_prompts='data/gsm8k/prompts/useful_examples.json',
               max_batch_size=2,
               max_response_length=200,
               mcts_rollouts=10,
               n_sample_subquestion=4,
               n_sample_confidence=8,
               temperature=0.8,
               max_depth=6,
               w_exp=1,
               r_alpha=0.5,
               r1_default=1,
               resume=0,
               log_dir=None,
               speedup_confidence_batch_size=None):
    pass


if __name__ == '__main__':
    fire.Fire(main_iters)