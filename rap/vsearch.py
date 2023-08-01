import io
import os
import re
from copy import deepcopy
import numpy as np
import torch
import tqdm
from tqdm import trange
from rap.models import QueryLM
from rap.utils.blocksworld import apply_change, generate_all_actions, get_world_change
from rap.iterative import ITERSNode, PITERS

import colorama
from colorama import Fore
from colorama import Style
colorama.init()


class ReasoningITERSNode(ITERSNode):
    @property
    def visited(self):
        return self._visited

    def __init__(self, prompt, gen_fn, reward_fn, depth, r1_default, r_alpha, max_depth=100, parent: 'ReasoningITERSNode' = None, r0=0., ):
        # self.n_sample = n_sample
        self._conf = None
        self.children = []
        self.prompt = prompt
        self.gen_fn = gen_fn
        self.reward_fn = reward_fn
        self.depth = depth
        self._r0 = r0
        self._r1 = self._r1_default = r1_default
        self._r_alpha = r_alpha
        self._ans_list = None
        self._visited = False
        self.parent = parent
        self.max_depth = max_depth

    def _child_node(self, prompt, r0):
        """Produce a child node object given its prompt and prior probability as r0"""
        return ReasoningITERSNode(prompt, self.gen_fn, self.reward_fn, self.depth + 1, self._r1_default, self._r_alpha, parent=self, r0=r0, max_depth=self.max_depth)

    def _get_children(self):
        # print("# in _get_children")
        self._visited = True
        self._calculate_reward()
        if self.is_terminal:
            return self.children
        questions, r0 = self.gen_fn(self.prompt, self.depth)
        for question, r in zip(questions, r0):
            self.children.append(self._child_node(question, r))
        return self.children

    def find_children(self):
        self.children = self.children or self._get_children()
        return self.children

    def _calculate_reward(self):
        # NOTE: temporary
        # print("# in _calculate_reward")
        # print("## depth", self.depth)
        if self.depth == 0:
            return
        self.prompt, self._r1, self._ans_list = self.reward_fn(self.prompt, self.depth)

    def _static_terminal(self):
        if self._r1 > 50:
            return True
        elif self.depth >= self.max_depth:
            return True
        else:
            return False

    @property
    def achieved_goal(self):
        return self._r1 > 50

    @property
    def is_terminal(self):
        return self._static_terminal() or self.reward < -1

    @property
    def reward(self):
        raise NotImplementedError
        # return self._r0 * self._r_alpha + self._r1 if self.depth >= self.max_depth else self._r0 * self._r_alpha
        # return self._r0 * self._r_alpha + self._r1

    def print(self, mcts, file=None):
        def pprint(*args):
            if file is None:
                tqdm.write(*args)
            else:
                print(*args, file=file)
        p1 = '-' * (4 * self.depth - 4)
        prefix = ' ' * (4 * self.depth - 4)
        question = self.prompt.split("[PLAN]\n")[-1].replace("\n", "\\n")
        pprint(p1 + question)
        pprint(prefix + f'R: {self.reward:.3f} ; N: {mcts.N[self]} ; M: {mcts.M[self]:.3f} ; r0 : {self._r0:.3f}')
        if not self.visited:
            return
        # answer = 'A' + self.prompt.split(f'Answer {self._prompt_index}')[-1].split('\n')[0]
        if self.reward < -1:
            if file is not None:
                pprint(prefix + question)
                # pprint(prefix + answer)
            return
        
        for child in self.children:
            child.print(mcts, file)
        if self.depth == 1:
            pprint("=" * 12)


def vicuna_search(initial_state: str,
                          goal: str,
                          prompts,
                          world_model: QueryLM,
                          temperature,
                          mcts_steps,
                          max_iter,
                          max_depth,
                          r_alpha,
                          r1_default,
                          eos_token_id,
                          speedup_action_batch_size=2,
                          w_exp=1,
                          sample_per_node=2,
                          search_depth=6):
    def gen_fn(inp, depth): # for r0=Pr(a|s_t), the probability component
        # print("# in gen_fn")
        last_state = re.search(f'.*{re.escape(prompts["state_prefix"].format(depth))}(.*)', inp)[1]
        # print("## input\n", inp)
        # print("## last state\n", last_state)

        raw_action_list = generate_all_actions(last_state)
        action_output = [inp + prompts["action_prefix"].format(depth + 1) + " " + a.capitalize() + ".\n" for a in raw_action_list]
        # print("========")
        # print("action list")
        new_action_output = []
        n_base_actions = 2 * (depth // 2)
        last_base_state = inp.split(prompts["state_prefix"].format(n_base_actions))[-1].split(prompts["action_prefix"].format(n_base_actions + 1))[0].strip()
        baseline_prompt = prompts["baseline_action"]
        baseline_prompt += "\n[STATEMENT]\n"
        baseline_prompt += "As initial conditions " + last_base_state.strip() + "\n" + inp.split("[GOAL]")[-1].split("[STATE 0]")[0].strip() + "\n\nMy plan is as follows:\n\n[PLAN]\n"

        action_list = []
        lastest_list = []
        for a in action_output:
            history = "".join(re.findall(r'\[ACTION \d+\] .*?\n', a)).replace(".", "")
            identifier = re.findall("\[ACTION \d+\]", history)
            for id in identifier:
                history = history.replace(id, "")
            history = history.strip().replace("\n ", "\n")
            torch.distributed.barrier()
            new_action_output.append(a)
            action_list.append(history)
            lastest_list.append("\n".join(history.split("\n")[-1 if depth % 2 == 0 else -2:]))
                
        # print("## action_list", action_list)
        # print("## last state in prompt: ", baseline_prompt.split("[STATE")[-1])

        ll_prompts = [baseline_prompt + a.lower() for a in lastest_list]
        
        # print("## evaluated actions in prompt: ", [prompt.split("[PLAN]\n")[-1] for prompt in ll_prompts])
        
        scores = []
        for idx in range(0, len(ll_prompts), speedup_action_batch_size):
            end_idx = min(idx + speedup_action_batch_size, len(ll_prompts))
            # TODO:
            with open('shit.txt', 'w') as f:
                f.write('ll_prompts='+str(ll_prompts[idx: end_idx])+'\n')
                f.write('baseline_prompt='+str([baseline_prompt]))
            if world_model.__class__.__name__ == 'QueryVicuna':
                log_probs = world_model.get_ll(baseline_prompt, ll_prompts[idx: end_idx])
            else:
                log_probs = world_model.llamamodel.get_ll(baseline_prompt, ll_prompts[idx: end_idx])
            scores += list(log_probs)
        # print("## log probs\n", scores)

        # softmax scores
        scores = np.array(scores)
        exp_scores = np.exp(scores)
        soft_scores = exp_scores / np.sum(exp_scores)
        # print("## soft scores\n", soft_scores)
        
        for a, s in zip(new_action_output, soft_scores):
            history = "".join(re.findall(r'\[ACTION \d+\].*?\n', a)).replace(".", "")
            identifier = re.findall("\[ACTION \d+\]", history)
            for id in identifier:
                history = history.replace(id, "")
            history = history.strip()
            # print("## action seq and score\n", history, s)

        return new_action_output, soft_scores
    
    def r1_fn(inp, depth): # for r1=Vrand

        # print("# in r1_fn")
        # print("## inp\n", inp)
        # print("## depth", depth)

        if depth == 0:
            return 1.0, inp, []
        
        last_state = re.search(f'.*{re.escape(prompts["state_prefix"].format(depth - 1))}(.*)', inp)[1]
        last_action = re.search(f'.*{re.escape(prompts["action_prefix"].format(depth))}(.*)', inp)[1]

        if "Pick" in last_action: 
            world_update_prompt = prompts["world_update_pickup"].format(last_state, last_action)
        elif "Unstack" in last_action:
            world_update_prompt = prompts["world_update_unstack"].format(last_state, last_action)
        elif "Put" in last_action:
            world_update_prompt = prompts["world_update_putdown"].format(last_state, last_action)
        elif "Stack" in last_action: 
            world_update_prompt = prompts["world_update_stack"].format(last_state, last_action)

        # world_output = world_model.query_LM(world_update_prompt, do_sample=False, num_return_sequences=1,
        #                             eos_token_id=eos_token_id)[0]
        # world_change = world_output.split("[CHANGE]")[-1]
        world_change = get_world_change(last_state, last_action)


        last_state = inp.split(f"[STATE {depth-1}]")[-1].split(f"[ACTION {depth}]")[0]
        # print("last state:\n", "\"" + last_state + "\"")
        new_state = apply_change(world_change, last_state)
        # print("==============new_state================")
        # print("\"" + new_state + "\"")
        new_prompt = inp + prompts["state_prefix"].format(depth) + " " + new_state + "\n"
        # print("new prompt:\n", "\"" + new_prompt + "\"")
        # print(world_change)
        goal_statement = inp.split("[GOAL]")[-1].split("[STATE 0]")[0]
        # print(f"goal_statement={goal_statement}")
        goals = re.findall("the [a-z]{0,10} block is on top of the [a-z]{0,10} block", goal_statement)
        # print(f"goals={goals}")
        meetings = [g in new_state for g in goals]
        # print(f"meetings={meetings}")
        if sum(meetings) == len(meetings):
            r1 = 100
        else:
            r1 = sum(meetings) / len(meetings) + 0.5
        return r1, new_prompt, []

    def reward_fn(inp, depth):
        # print("# in reward_fn")
        r1, answer, ans_list = r1_fn(inp, depth)
        return answer, r1, ans_list

    # TODO: debug
    # print(f'{Fore.YELLOW}Start a task.{Style.RESET_ALL}')
    # print(f'{Fore.YELLOW}In this task, the initial state is::::: {initial_state} :::::{Style.RESET_ALL}')
    # print(f'{Fore.YELLOW}In this task, the goal is::::: {goal} :::::{Style.RESET_ALL}')
    # print(f'{Fore.YELLOW}In this task, the prompts partially include {Style.RESET_ALL}::::: \
    #     question_prefix={prompts["question_prefix"]} \
    #     state_prefix={prompts["state_prefix"]} \
    #     goal_prefix={prompts["goal_prefix"]} \
    #     action_prefix={prompts["action_prefix"]} \
    #     action_gen_prefix={prompts["action_gen_prefix"]} \
    #     action_reason_prefix={prompts["action_reason_prefix"]} \
    #     state_gen_prefix={prompts["state_gen_prefix"]} \
    #     confidence_prefix={prompts["confidence_prefix"]} \
    #     confidence_answer_prefix={prompts["confidence_answer_prefix"]} \
    #     validity_prefix={prompts["validity_prefix"]}:::::')
    # print(f'{Fore.YELLOW}In this task, the maximum iteration step is ::::: {max_iter} :::::{Style.RESET_ALL}')
    # print(f'{Fore.YELLOW}In this task, the maximum lookahead step is ::::: {max_depth} :::::{Style.RESET_ALL}')

    its = PITERS(w_exp=w_exp, prior=True, aggr_reward='mean', aggr_child='max', sample_per_node=sample_per_node, search_depth=search_depth)
    start_node = ReasoningITERSNode(prompts["goal_prefix"] + goal.strip() + "\n" + prompts["state_prefix"].format(0) + " " + initial_state.strip() + "\n", gen_fn, reward_fn, depth=0, r1_default=r1_default, r_alpha=r_alpha, max_depth=max_depth)

    # TODO: debug
    # print(f'{Fore.YELLOW}Created start_node. Its current ::::: reward={start_node.reward}, consisting of r0={start_node._r0} and r1={start_node._r1}. Num of children={len(start_node.children)}. Has parent? answer={start_node.parent is None} :::::{Style.RESET_ALL}')

    trajs = []
    iters = []
    
    # TODO: debug
    # print(f'{Fore.YELLOW}Start a loop. ::::: Total loop range={mcts_steps} :::::{Style.RESET_ALL}')
    for _ in (pbar := trange(mcts_steps, disable=bool(int(os.environ.get("LOCAL_RANK", -1))), position=0)):
        end_node = its.rollout(max_iter, start_node)
        # print(f'{Fore.BLUE}Got an end node. Its prompt is {Style.RESET_ALL}:::::{end_node.prompt}:::::. {Fore.BLUE}This prompt will be a "traj". Its depth={end_node.depth}. Its reward={start_node.reward}, consisting of r0={start_node._r0} and r1={start_node._r1}. Num of children={len(start_node.children)}. Has parent? answer={start_node.parent is None} :::::{Style.RESET_ALL}')
        trajs.append(traj := end_node.prompt)
        output = re.findall('The answer is .*?([.0-9,\\-]+).*\\.', traj)
        # print(f'{Fore.BLUE}From the traj, our output is {Style.RESET_ALL}:::::{output}:::::{Fore.BLUE}If the output has len 0 ({len(output) == 0}), then we mark it as "not found"{Style.RESET_ALL}')
        if len(output) == 0:
            temp_r = 'not found'
        else:
            temp_r = output[-1].replace(',', '')
        pbar.set_description(f'{temp_r}')
        iter_copy = deepcopy(start_node)
        iter_copy.Q = dict(its.Q)
        iter_copy.N = dict(its.N)
        iter_copy.M = dict(its.M)
        iters.append(iter_copy)

    with io.StringIO() as f:
        start_node.print(its, file=f)
        tree = f.getvalue()
    return trajs, tree, iters