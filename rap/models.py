from abc import ABC, abstractmethod

from typing import List

import uuid

import torch

from llama import LLaMA

from fastchat.model import load_model
from fastchat.serve.model_worker import ModelWorker


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


class QueryVicuna(QueryLM):
    def __init__(self, model_path='lmsys/vicuna-7b-v1.3', num_gpus=1) -> None:
        self.llamamodel, self.tokenizer = load_model(
            model_path=model_path,
            device='cuda',
            num_gpus=num_gpus,
            max_gpu_memory='40GiB',
        )
        self.tokenizer.eos_id = self.tokenizer.encode('\n')[0]

    def query_LM(self):
        raise NotImplementedError

    def query_next_token(self):
        raise NotImplementedError
    
    @torch.no_grad()
    def get_ll(
        self,
        prefix: str,
        prompts: List[str],
    ) -> List[str]:
        bsz = len(prompts)
        prefix_tokens = self.tokenizer(prefix, return_tensors="pt")
        prompts_tokens = [self.tokenizer(x, return_tensors="pt") for x in prompts]
        max_prompt_size = max([len(t.input_ids[0]) for t in prompts_tokens])
        total_len = max_prompt_size
        tokens = torch.full((bsz, total_len), self.tokenizer.eos_id).cuda().long()

        logits = []
        for k, t in enumerate(prompts_tokens):
            tokens[k, : len(t.input_ids[0])] = torch.tensor(t.input_ids)[:self.tokenizer.model_max_length].long()
            logits.append(self.llamamodel(tokens[k:k+1, :].to(self.llamamodel.device)).logits)

    #   logits = self.model(tokens.to(self.model.device)).logits
        logits = torch.cat(logits, dim=0)
        acc_probs = torch.zeros(bsz).to(self.llamamodel.device)
        for i in range(len(prefix_tokens.input_ids[0]), max_prompt_size):
            probs = torch.softmax(logits[:, i - 1, :], dim=-1)
            for j in range(bsz):    
                if tokens[j, i] != self.tokenizer.eos_id:
                    acc_probs[j] += torch.log(probs[j, tokens[j, i]])

        return acc_probs.cpu().numpy()


class QueryVicunaAuto(QueryVicuna):
    def __init__(self, controller_addr, worker_address, model_path='lmsys/vicuna-7b-v1.3', num_gpus=1) -> None:
        worker_id = str(uuid.uuid4())[:8]
        self.modelworker = ModelWorker(
            controller_addr,
            worker_address,
            worker_id,
            model_path,
            model_names=None,
            limit_worker_concurrency=5,
            no_register=False,
            device="cuda",
            num_gpus=num_gpus,
            max_gpu_memory='40GiB',
        )
        self.llamamodel, self.tokenizer = self.modelworker.model, self.modelworker.tokenizer
        self.tokenizer.eos_id = self.tokenizer.encode('\n')[0]
