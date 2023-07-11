# 1. copy paste from run_blocksworld.py
# 2. Replace rap.blocksworld_mcts with rap.mybw_mcts

import os
import yaml
import sys
sys.path.append("gpt-plan-benchmark/gpt_plan_test")
from Executor import Executor
from utils import *
from pathlib import Path
from tarski.io import PDDLReader
import argparse
import time
import random
import numpy as np

from rap.psearch import policy_search
from rap.models import QueryLlama

import torch
from llama import *
from typing import Tuple
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
import json
import time
import re
import pickle

import colorama
from colorama import Fore
from colorama import Style
colorama.init()


def validate_plan(domain, instance, plan_file):
    val_path = os.getenv("VAL")
    cmd = f"{val_path}/validate {domain} {instance} {plan_file}"
    response = os.popen(cmd).read()

    # print("RESPONSE:::", response)
    if 'Problem in domain' in response:
        raise Exception('Problem in domain: Check PDDL Writer')

    if "Plan valid" in response:
        return True, response
    else:
        return False, response


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    # torch.manual_seed(1)
    return local_rank, world_size

def load(ckpt_dir: str, tokenizer_path: str, local_rank: int, world_size: int, max_batch_size: int) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    # print(checkpoints)
    assert (
            world_size == len(checkpoints)
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    # print("Loading")
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
    # print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator

success_template = "{} {} {} {}"
verbose_template="""
{}
--------- LLM response ---------
{}
--------- Extracted plan ---------
{}
-------- Ground truth plan ---------
{}
{}
"""

class ReasoningTasks():

    def __init__(self, verbose=False, model_name="LLaMA", ckpt_path="", data_path=""):
        # self.engine = engine
        self.verbose = verbose
        self.max_gpt_response_length = 500
        self.data_files = json.load(open(data_path, 'r'))
        self.model_name = model_name

        self.plan_file = "sas_plan"
        self.lm_plan_file = "gpt_sas_plan"

        if local_rank > 0:
            sys.stdout = open(os.devnull, 'w')
            log_file = None
        else:
            log_file = "logs/interactive.log"

        self.local_rank = local_rank

        if self.model_name == "LLaMA":
            llm = ckpt_path
            # the parent directory of the checkpoint directory
            tokenizer_path = os.path.join(os.path.dirname(llm), "tokenizer.model")
            # print(tokenizer_path)
            llama = load(llm, tokenizer_path, local_rank, world_size, 3)
            self.model = QueryLlama(llama, max_response_length=100, log_file=log_file)
        else:
            raise NotImplementedError
        

    # ========================================== UTILS ========================================== #
    def compute_plan(self, domain, instance, timeout=30):
        fast_downward_path = os.getenv("FAST_DOWNWARD")
        # Remove > /dev/null to see the output of fast-downward
        assert os.path.exists(f"{fast_downward_path}/fast-downward.py")
        
        if local_rank == 0:
            if os.path.exists(self.plan_file):
                try:
                    os.remove(self.plan_file)
                except Exception as e:
                    print(e)

            while not os.path.exists(self.plan_file):
                cmd = f"timeout {timeout}s {fast_downward_path}/fast-downward.py --log-level debug {domain} {instance} --search \"astar(lmcut())\"  > /dev/null 2>&1"
                os.system(cmd)
                time.sleep(2)
                
        torch.distributed.barrier()
        
        if not os.path.exists(self.plan_file):
            # print("Plan failed")
            return ""
        
        return Path(self.plan_file).read_text()

    def read_config(self, config_file):
        with open(config_file, 'r') as file:
            self.data = yaml.safe_load(file)

    def get_problem(self, instance, domain):
        reader = PDDLReader(raise_on_error=True)
        reader.parse_domain(domain)
        return reader.parse_instance(instance)

    def get_executor(self, instance, domain):
        plan_executor = Executor(domain, instance)
        return plan_executor

    def save_output(self, output_file, final_output):
        os.makedirs(f"outputs/{self.model_name}/", exist_ok=True)
        with open(f"outputs/{self.model_name}/" + output_file + ".txt", 'w+') as f:
            f.write(final_output)
    # ========================================== TASKS ========================================== #

    def run_mcts(self, config_file, name="", prompts="", rollouts=10, max_iter=30, max_depth=4, alpha=0.5, prompt_path="", resume_file_idx=0, sample_per_node=2):
        self.read_config(config_file)

        # make directory for logs
        os.makedirs(f"logs/mcts-{name}/json/", exist_ok=True)
        os.makedirs(f"logs/mcts-{name}/tree/", exist_ok=True)
        os.makedirs(f"logs/mcts-{name}/pkl/", exist_ok=True)

        n_files = len(self.data_files)
        domain_pddl = f'gpt-plan-benchmark/gpt_plan_test/instances/{self.data["domain_file"]}'

        final_output = ""
        correct_plans = 0

        if local_rank == 0:
            if os.path.exists(self.plan_file):
                os.remove(self.plan_file)
            if os.path.exists(self.lm_plan_file):
                os.remove(self.lm_plan_file)
        

        with open(prompt_path) as f:
            prompts = json.load(f)
            '''
            "prompts" here is a dictionary.
            
            It has keys: ['action_generation', 'world_update', 'world_update_pickup', 'world_update_unstack', 'world_update_putdown', 'world_update_stack', 'confidence', 'action_validity_pickup', 'action_validity_unstack', 'action_validity_putdown', 'action_validity_stack', 'complete_validity', 'state_extract', 'hand_state_extract', 'question_prefix', 'state_prefix', 'goal_prefix', 'action_prefix', 'action_gen_prefix', 'action_reason_prefix', 'state_gen_prefix', 'confidence_prefix', 'confidence_answer_prefix', 'validity_prefix', 'complete_validity_prefix', 'baseline_action']
            
            One can refer to ./data/blocksworld/my_mcts_prompts_update.json for an example of prompts.

            It has several components and functionality:
            1. It gives the llm few shot examples on the task. It tells llm the action-state transition rules.
            2. Log the state, action ... at this node. Relevant keys are:
                "question_prefix": "[SCENARIO 2]\n",
                "state_prefix": "[STATE {}]",
                "goal_prefix": "[GOAL] ",
                "action_prefix": "[ACTION {}]",
                "action_gen_prefix":  "[ACTIONS]",
                "action_reason_prefix":  "[REASON]",
                "state_gen_prefix":  "[STATE 1]",
                "confidence_prefix": "[QUESTION] Is \"STATE 1\" closer to the goal than \"STATE 0\"?",
                "confidence_answer_prefix": "[ANSWER]",
                "validity_prefix": "[QUESTION] Is the action valid based on the state?",            
            '''

        mcts_steps = rollouts
        total_correct = [0] * mcts_steps
        # TODO: debug
        print(f'{Fore.YELLOW}There are {n_files} files.{Style.RESET_ALL}')
        for i in range(n_files):
            # TODO: debug
            print(f'{Fore.YELLOW}We are dealing with {i}-th file now.{Style.RESET_ALL}')
            if i < resume_file_idx:
                if self.local_rank == 0:
                    correct_plans += 1
                continue

            # query = prompts
            cur_instance = self.data_files[i]
            problem = self.get_problem(cur_instance[0], domain_pddl)
            gt_plan_text = cur_instance[1]
            INIT, GOAL, PLAN = instance_to_text_blocksworld(problem, False, self.data)

            query = prompts["baseline_action"]
            # gt_plan = self.compute_plan(domain_pddl, cur_instance)
            query += fill_template(*instance_to_text_blocksworld(problem, False, self.data)) + "\n"
            
            trajs, tree, trees = policy_search(
                f'I have that, {INIT}.', 
                f'My goal is to have that {GOAL}.',
                prompts, 
                self.model, 
                temperature=0.6,
                mcts_steps=mcts_steps,
                max_iter=max_iter,
                max_depth=max_depth,
                r1_default=0.5,
                eos_token_id=self.model.tokenizer.encode('\n', bos=False, eos=False)[-1],
                r_alpha=alpha,
                sample_per_node=sample_per_node
                )

            torch.distributed.barrier()

            if self.local_rank == 0:
                json_logs = []
                for rollout, traj in enumerate(trajs):
                    # print("evaluating one rollout")
                    # Extract actions from trace
                    # actions = re.findall('\[ACTION \d\](.*)', traj)
                    # Do text_to_plan procedure
                    actions = re.findall('\[ACTION \d\](.*)', traj)
                    _, lm_plan = text_to_plan_blocksworld('\n'.join(actions), problem.actions, self.lm_plan_file, self.data)
                    # Apply VAL
                    correct, response = validate_plan(domain_pddl, cur_instance[0], self.lm_plan_file)

                    json_logs.append({
                        'rollout': rollout + 1,
                        'initial_state': INIT,
                        'goal': GOAL,
                        'output': response,
                        'correct': correct,
                        'traj': traj,
                    })
                    total_correct[rollout] += correct
                with open(os.path.join(f'./logs/mcts-{name}/json/', f'{i:04d}.json'), 'w') as f:
                    json.dump(json_logs, f, indent=2)
                with open(os.path.join(f'./logs/mcts-{name}/tree/', f'{i:04d}.tree'), 'w') as f:
                    f.write(tree)
                # with open(os.path.join(f'./logs/mcts-{name}/pkl/', f'{i:04d}.pkl'), 'wb') as f:
                #     pickle.dump(trees, f)


            torch.distributed.barrier()
            actions = re.findall('\[ACTION \d\](.*)', trajs[-1])
            _, lm_plan = text_to_plan_blocksworld('\n'.join(actions), problem.actions, self.lm_plan_file, self.data)

            correct, response = validate_plan(domain_pddl, cur_instance[0], self.lm_plan_file)
            correct_plans += int(correct)

            final_output += success_template.format('='*35, "MCTS", "SUCCESS" if correct else "FAILURE", '='*35)
            final_output += response
            final_output += verbose_template.format(f'I have that, {INIT}\n My goal is to have that {GOAL}', trajs[-1], lm_plan, gt_plan_text, '='*77) if self.verbose else ""
            # if self.verbose: print(final_output)

            self.save_output("mcts-" + name, final_output)

        if local_rank == 0:
            if os.path.exists(self.plan_file):
                os.remove(self.plan_file)
            if os.path.exists(self.lm_plan_file):
                os.remove(self.lm_plan_file)

        # --------------- Add to final output --------------- #
        final_output += f"[+]: The number of correct plans is {correct_plans}/{n_files}={correct_plans / (n_files) * 100}%"
        print(f"[+]: The number of correct plans is {correct_plans}/{n_files}={correct_plans / (n_files) * 100}%")
        print(total_correct)
        self.save_output("mcts-" + name, final_output)

if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    local_rank, world_size = setup_model_parallel()

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='mcts', help='Task to run t1 = Goal Directed Reasoning')
    parser.add_argument('--model_name', type=str, default='LLaMA', help='Model to use')
    parser.add_argument('--verbose', type=str, default="False", help='Verbose')
    parser.add_argument('--name', type=str, default="unnamed", help='Name of the experiment')
    parser.add_argument('--data_path', type=str, default="data", help='Path to data')
    parser.add_argument('--rollouts', type=int, default=10, help='Number of rollouts')
    parser.add_argument('--max_iter', type=int, default=30)
    parser.add_argument('--max_depth', type=int, default=4, help='Max depth of the tree')
    parser.add_argument('--alpha', type=float, default=0.5, help='Alpha for reward')
    parser.add_argument('--n_samples', type=int, default=10, help='Number of samples for t1')
    parser.add_argument('--prompt_path', type=str, default="data/blocksworld/my_mcts_prompts_update.json", help='Path to prompts')
    parser.add_argument('--ckpt_path', type=str, default="", help='path to LLaMA checkpoint')
    parser.add_argument('--resume_file_idx', type=int, default=0, help='resume experiment from a certain task')
    parser.add_argument('--sample_per_node', type=int, default=2, help='number of samples we take in the lookahead trajectory search')


    args = parser.parse_args()
    task = args.task
    model_name = args.model_name
    data_path = args.data_path
    rollouts = args.rollouts
    alpha = args.alpha
    n_samples = args.n_samples
    # engine = args.engine
    name = args.name
    max_depth = args.max_depth
    verbose = eval(args.verbose)
    prompt_path = args.prompt_path
    ckpt_path = args.ckpt_path
    max_iter = args.max_iter

    tasks_obj = ReasoningTasks(verbose, model_name=model_name, data_path=data_path, ckpt_path=ckpt_path)

    if task == 'mcts':
        config_file = 'data/blocksworld/bw_config.yaml'
        tasks_obj.run_mcts(config_file, name=name, prompts="", rollouts=rollouts, max_iter=max_iter, max_depth=max_depth, alpha=alpha, prompt_path=prompt_path, resume_file_idx=args.resume_file_idx, sample_per_node=args.sample_per_node)
    else:
        raise NotImplementedError