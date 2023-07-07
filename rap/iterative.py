import io
import json
import os
import math
import random
import re
import sys
import traceback
import warnings
from collections import defaultdict
from abc import ABC, abstractmethod
from copy import deepcopy
import torch
from rap.utils.blocksworld import apply_change, generate_all_actions

sys.path.append("gpt-plan-benchmark/gpt_plan_test")
from utils import *
from rap.models import QueryLM
from tqdm import tqdm, trange
import numpy as np


def validate_plan(domain, instance, plan_file):
    val_path = os.getenv("VAL")
    cmd = f"{val_path}/validate {domain} {instance} {plan_file}"
    response = os.popen(cmd).read()

    print("RESPONSE:::", response)
    if 'Problem in domain' in response:
        raise Exception('Problem in domain: Check PDDL Writer')

    if "Plan valid" in response:
        return True, response
    else:
        return False, response


class ITERSNode(ABC):
    @abstractmethod
    def find_children(self):
        return set()

    @property
    @abstractmethod
    def is_terminal(self):
        return True

    @property
    @abstractmethod
    def reward(self):
        return 0

    @property
    @abstractmethod
    def visited(self):
        return 0


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
        print("# in _get_children")
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
        print("# in _calculate_reward")
        print("## depth", self.depth)
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
        return self._r0 * self._r_alpha + self._r1


class ITERS:
    """Iterative lookahead planning with llm"""
    def __init__(self, w_exp=1, discount=1, prior=False, aggr_reward='sum', aggr_child='max', problem=None):
        self.Q: dict[ITERSNode, float] = defaultdict(lambda : 0.)
        self.N: dict[ITERSNode, int] = defaultdict(lambda : 0)
        self.M: dict[ITERSNode, float] = defaultdict(lambda : -math.inf)
        self.children = dict()
        self.w_exp = w_exp
        self.discount = discount
        self.prior = prior
        self.aggr_reward = aggr_reward
        self.aggr_child = aggr_child
        self.problem = problem

    def rollout(self, max_iter: int, node: ITERSNode):
        for k in range(max_iter):
            # generate candidate lookahead paths
            paths = self._lookahead(node)
            # calculate the return from each path
            for path in paths:
                self._back_propagate(path)
            # choose the path with maximum return
            max_path = self._max_ahead(paths)
            # the next node is the ending node of the chosen path
            next_node = max_path[-1]
            # stop iteration if we reached the goal
            if next_node.achieved_goal:
                break
        return next_node

    def _back_propagate(self, path: list[ITERSNode], reward=0.):
        coeff = 1
        for node in reversed(path):
            reward = reward * self.discount + node.reward
            coeff = coeff * self.discount + 1
            if self.aggr_reward == 'mean':
                c_reward = reward / coeff
            else:
                c_reward = reward
            if node not in self.N:
                self.Q[node] = c_reward
            else:
                self.Q[node] += c_reward
            self.N[node] += 1
            self.M[node] = max(self.M[node], c_reward)

    def _lookahead(self, node: ITERSNode):
        paths = []
        def route(node, path):
            self._expand(node)
            if node.is_terminal:
                paths.append(path)
            for new_node in self.children[node]:
                path.append(new_node)
                route(new_node, path)
        route(node, [])
        return paths

    def _max_ahead(self, paths: list[list[ITERSNode]]):
        return max(paths, key=lambda x: self.M[x[0]])


def iterative_search(initial_state: str,
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
                          problem,
                          speedup_action_batch_size=2,
                          w_exp=1):
    def gen_fn(inp, depth): # for r0=Pr(a|s_t), the probability component
        print("# in gen_fn")
        last_state = re.search(f'.*{re.escape(prompts["state_prefix"].format(depth))}(.*)', inp)[1]
        print("## input\n", inp)
        print("## last state\n", last_state)

        raw_action_list = generate_all_actions(last_state)
        action_output = [inp + prompts["action_prefix"].format(depth + 1) + " " + a.capitalize() + ".\n" for a in raw_action_list]
        print("========")
        print("action list")
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
                
        print("## action_list", action_list)
        print("## last state in prompt: ", baseline_prompt.split("[STATE")[-1])

        ll_prompts = [baseline_prompt + a.lower() for a in lastest_list]
        
        print("## evaluated actions in prompt: ", [prompt.split("[PLAN]\n")[-1] for prompt in ll_prompts])
        
        scores = []
        for idx in range(0, len(ll_prompts), speedup_action_batch_size):
            end_idx = min(idx + speedup_action_batch_size, len(ll_prompts))
            log_probs = world_model.llamamodel.get_ll(baseline_prompt, ll_prompts[idx: end_idx])
            scores += list(log_probs)
        print("## log probs\n", scores)

        # softmax scores
        scores = np.array(scores)
        exp_scores = np.exp(scores)
        soft_scores = exp_scores / np.sum(exp_scores)
        print("## soft scores\n", soft_scores)
        
        for a, s in zip(new_action_output, soft_scores):
            history = "".join(re.findall(r'\[ACTION \d+\].*?\n', a)).replace(".", "")
            identifier = re.findall("\[ACTION \d+\]", history)
            for id in identifier:
                history = history.replace(id, "")
            history = history.strip()
            print("## action seq and score\n", history, s)

        return new_action_output, soft_scores
    
    def r1_fn(inp, depth): # for r1=Vrand

        print("# in r1_fn")
        print("## inp\n", inp)
        print("## depth", depth)

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

        world_output = world_model.query_LM(world_update_prompt, do_sample=False, num_return_sequences=1,
                                    eos_token_id=eos_token_id)[0]


        world_change = world_output.split("[CHANGE]")[-1]
        # print("world change:\n" + "\"" + world_change + "\"")     
        # print("==============inp================")
        # print(inp)
        last_state = inp.split(f"[STATE {depth-1}]")[-1].split(f"[ACTION {depth}]")[0]
        print("last state:\n", "\"" + last_state + "\"")
        new_state = apply_change(world_change, last_state)
        print("==============new_state================")
        print("\"" + new_state + "\"")
        new_prompt = inp + prompts["state_prefix"].format(depth) + " " + new_state + "\n"
        print("new prompt:\n", "\"" + new_prompt + "\"")
        # print(world_change)
        goal_statement = inp.split("[GOAL]")[-1].split("[STATE 0]")[0]
        print(f"goal_statement={goal_statement}")
        goals = re.findall("the [a-z]{0,10} block is on top of the [a-z]{0,10} block", goal_statement)
        print(f"goals={goals}")
        meetings = [g in new_state for g in goals]
        print(f"meetings={meetings}")
        if sum(meetings) == len(meetings):
            r1 = 100
        else:
            r1 = sum(meetings) / len(meetings) + 0.5
        return r1, new_prompt, []

    def reward_fn(inp, depth):
        print("# in reward_fn")
        r1, answer, ans_list = r1_fn(inp, depth)
        return answer, r1, ans_list

    its = ITERS(w_exp=w_exp, prior=True, aggr_reward='mean', aggr_child='max', problem=problem)
    start_node = ReasoningITERSNode(prompts["goal_prefix"] + goal.strip() + "\n" + prompts["state_prefix"].format(0) + " " + initial_state.strip() + "\n", gen_fn, reward_fn, depth=0, r1_default=r1_default, r_alpha=r_alpha, max_depth=max_depth)
    trajs = []
    iters = []
    for _ in (pbar := trange(mcts_steps, disable=bool(int(os.environ.get("LOCAL_RANK", -1))), position=0)):
        end_node = its.rollout(max_iter, start_node)
        trajs.append(traj := end_node.prompt)
        output = re.findall('The answer is .*?([.0-9,\\-]+).*\\.', traj)
        if len(output) == 0:
            temp_r = 'not found'
        else:
            temp_r = output[-1].replace(',', '')
        iter_copy = deepcopy(start_node)
        iter_copy.Q = dict(its.Q)
        iter_copy.N = dict(its.N)
        iter_copy.M = dict(its.M)
        iters.append(iter_copy)

    with io.StringIO() as f:
        start_node.print(its, file=f)
        tree = f.getvalue()
    return trajs, tree, iters