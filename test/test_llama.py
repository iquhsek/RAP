import os
import sys
sys.path.append('../')
import json
import argparse
from pathlib import Path
from typing import Tuple
from abc import ABC, abstractmethod
import torch
from llama import LLaMA, ModelArgs, Transformer, Tokenizer
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
import colorama
from colorama import Fore
from colorama import Style
colorama.init()


class QueryLM(ABC):
    @abstractmethod
    def query_LM(self, prompt, **gen_kwargs):
        pass

    @abstractmethod
    def query_next_token(self, prompt: list[str]):
        pass


class QueryHfModel(QueryLM):
    # This is not well-tested. Please use LLaMA if possible.
    def query_next_token(self, prompt: list[str]):
        raise NotImplementedError

    def __init__(self, model, tokenizer, max_response_length, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.n_examples = 1
        self.max_response_length = max_response_length

    def query_LM(self, prompt, **gen_kwargs):
        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            # print("input length", len(inputs))
            # Generate
            generate_ids = self.model.generate(inputs.input_ids, max_new_tokens=self.max_response_length, **gen_kwargs)
            text = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return text


class QueryLlama(QueryLM):
    def __init__(self, llamamodel: LLaMA, max_response_length, log_file) -> None:
        self.llamamodel = llamamodel
        self.tokenizer = self.llamamodel.tokenizer
        self.max_response_length = max_response_length
        self.log_file = log_file
        self.max_batch_size = llamamodel.model.params.max_batch_size
        self.yes_no = self.tokenizer.encode('Yes No', bos=False, eos=False)

    def query_LM(self, prompt, eos_token_id, num_return_sequences=1, do_sample=True, temperature=0.8):
        temperature = temperature if do_sample else 0
        all_results = []
        for start in range(0, num_return_sequences, self.max_batch_size):
            end = min(start + self.max_batch_size, num_return_sequences)
            results = self.llamamodel.generate([prompt] * (end - start), max_gen_len=self.max_response_length, temperature=temperature, eos_token_id=eos_token_id)
            all_results.extend(results)
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write("="*50+"\n")
                f.write(prompt + "\n")
                for result in all_results:
                    f.write("-"*50+"\n")
                    f.write(result.replace(prompt, "") + "\n")
        return all_results

    @torch.no_grad()
    def query_next_token(self, prompts):
        if isinstance(prompts, str):
            prompts = [prompts]
        ret = []
        for prompt in prompts:
            tokens = self.tokenizer.encode(prompt, bos=True, eos=False)
            tokens = torch.tensor([tokens]).cuda().long()
            output, h = self.llamamodel.model.forward(tokens, start_pos=0)
            ret.append(output)
        outputs = torch.cat(ret, dim=0)
        filtered = outputs[:, self.yes_no]
        dist = torch.softmax(filtered, dim=-1)
        return dist


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # Seed must be the same in all processes
    # torch.manual_seed(1)
    return local_rank, world_size


def load(ckpt_dir: str, tokenizer_path: str, local_rank: int, world_size: int, max_batch_size: int) -> LLaMA:
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert (
            world_size == len(checkpoints)
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
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
    return generator


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default="", help='path to LLaMA checkpoint')
    args = parser.parse_args()
    local_rank, world_size = setup_model_parallel()
    tokenizer_path = os.path.join(os.path.dirname(args.ckpt_path), "tokenizer.model")
    llama = load(args.ckpt_path, tokenizer_path, local_rank, world_size, 3)
    if local_rank > 0:
        sys.stdout = open(os.devnull, 'w')
        log_file = None
    else:
        log_file = "../logs/test_llama.log"
    # Query agent by supported LLaMA
    world_model = QueryLlama(llama, max_response_length=100, log_file=log_file)
    # Encode
    eos_token_id= world_model.tokenizer.encode('\n', bos=False, eos=False)[-1]
    
    
    # API 1: world_model.query_LM
    prompts = json.load('llama_prompts.json')
    last_state = "I have that, the blue block is clear, the orange block is clear, the yellow block is clear, the hand is empty, the orange block is on top of the red block, the blue block is on the table, the red block is on the table, and the yellow block is on the table."
    last_action = "Pick up the blue block."
    if "Pick" in last_action: 
            world_update_prompt = prompts["world_update_pickup"].format(last_state, last_action)
    elif "Unstack" in last_action:
        world_update_prompt = prompts["world_update_unstack"].format(last_state, last_action)
    elif "Put" in last_action:
        world_update_prompt = prompts["world_update_putdown"].format(last_state, last_action)
    elif "Stack" in last_action: 
        world_update_prompt = prompts["world_update_stack"].format(last_state, last_action)
    world_output = world_model.query_LM(
        world_update_prompt,
        do_sample=False,
        num_return_sequences=1,
        eos_token_id=eos_token_id)[0]
    print(f'world_output={world_output}')
    
    
    # API 2: world_model.llamamodel.get_ll
    last_base_state = prompts["last_base_state"]
    baseline_prompt = prompts["baseline_action"]
    log_probs = world_model.llamamodel.get_ll(baseline_prompt, last_base_state)
    print(f'log_probs={log_probs}')
