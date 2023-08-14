import re
import torch
import numpy as np
from copy import deepcopy
from typing import List, Tuple
from rap.forward_search import ForwardSearch, AbsNode
from rap.utils.blocksworld import apply_change, generate_all_actions, get_world_change


class StateNode(AbsNode):
    def __init__(self, prompt, rwd_fn, v_fn, alpha, depth, max_depth, parent=None, prob_r=0):
        self.children = []
        self.depth = depth # real depth, not relative depth
        self.max_depth = max_depth
        self.parent = parent
        self._alpha = alpha
        self.rwd_fn = rwd_fn
        self.v_fn = v_fn
        self._prob_r = prob_r
        self.prompt, self._v_rand = v_fn(prompt, depth)

    def _create_child_node(self, prompt, prob_r):
        return StateNode(prompt, self.rwd_fn, self.v_fn, self._alpha, self.depth + 1, self.max_depth, parent=self, prob_r=prob_r)

    def _get_children(self):
        if self.is_terminal:
            return self.children
        questions, arr_prob_r = self.rwd_fn(self.prompt, self.depth)
        for question, prob_r in zip(questions, arr_prob_r):
            child = self._create_child_node(question, prob_r)
            if child.achieved_goal:
                return [child]
            self.children.append(child)
        return self.children

    def get_children(self):
        self.children = self.children or self._get_children()
        return self.children

    @property
    def achieved_goal(self) -> bool:
        return self._v_rand > 100

    @property
    def is_terminal(self) -> bool:
        return self.depth > self.max_depth or self.achieved_goal


def forward_plan(initial_state: str,
                 goal: str,
                 prompts: dict,
                 world_model,
                 alpha: float,
                 horizon: int,
                 search_depth: int,
                 sample_per_node: int,
                 sampler: str='heuristic',
                 discount: float=1,
                 speedup_action_batch_size=2,) -> bool:

    '''-----------------------------------------------------'''

    def rwd_fn(inp, depth) -> Tuple[List, np.ndarray]:
        '''For r=Pr(a|s_t), the probability component reward'''
        last_state = re.search(f'.*{re.escape(prompts["state_prefix"].format(depth))}(.*)', inp)[1]
        raw_action_list = generate_all_actions(last_state)
        action_output = [inp + prompts["action_prefix"].format(depth + 1) + " " + a.capitalize() + ".\n" for a in raw_action_list]
        n_base_actions = 2 * (depth // 2)
        last_base_state = inp.split(prompts["state_prefix"].format(n_base_actions))[-1].split(prompts["action_prefix"].format(n_base_actions + 1))[0].strip()
        baseline_prompt = prompts["baseline_action"]
        baseline_prompt += "\n[STATEMENT]\n"
        baseline_prompt += "As initial conditions " + last_base_state.strip() + "\n" + inp.split("[GOAL]")[-1].split("[STATE 0]")[0].strip() + "\n\nMy plan is as follows:\n\n[PLAN]\n"
        action_list = []
        lastest_list = []
        new_action_output = []
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
        ll_prompts = [baseline_prompt + a.lower() for a in lastest_list]
        scores = []
        for idx in range(0, len(ll_prompts), speedup_action_batch_size):
            end_idx = min(idx + speedup_action_batch_size, len(ll_prompts))
            if world_model.__class__.__name__ == 'QueryVicuna':
                log_probs = world_model.get_ll(baseline_prompt, ll_prompts[idx: end_idx])
            else:
                log_probs = world_model.llamamodel.get_ll(baseline_prompt, ll_prompts[idx: end_idx])
            scores += list(log_probs)
        scores = np.array(scores)
        exp_scores = np.exp(scores)
        soft_scores = exp_scores / np.sum(exp_scores)
        return new_action_output, soft_scores

    '''-----------------------------------------------------'''

    def v_fn(inp, depth) -> Tuple[str, float]:
        if depth == 0:
            return inp, 0
        last_state = re.search(f'.*{re.escape(prompts["state_prefix"].format(depth - 1))}(.*)', inp)[1]
        last_action = re.search(f'.*{re.escape(prompts["action_prefix"].format(depth))}(.*)', inp)[1]
        world_change = get_world_change(last_state, last_action)
        last_state = inp.split(f"[STATE {depth-1}]")[-1].split(f"[ACTION {depth}]")[0]
        new_state = apply_change(world_change, last_state)
        new_prompt = inp + prompts["state_prefix"].format(depth) + " " + new_state + "\n"
        goal_statement = inp.split("[GOAL]")[-1].split("[STATE 0]")[0]
        goals = re.findall("the [a-z]{0,10} block is on top of the [a-z]{0,10} block", goal_statement)
        meetings = [g in new_state for g in goals]
        v = 200 if sum(meetings) == len(meetings)\
            else sum(meetings) / len(meetings) + 0.5
        return new_prompt, v

    '''-----------------------------------------------------'''

    planner = ForwardSearch(
        search_depth=search_depth,
        sample_per_node=sample_per_node,
        sampler=sampler,
        discount=discount
    )
    cur_node = StateNode(
        prompt=prompts["goal_prefix"] + goal.strip() + "\n" + prompts["state_prefix"].format(0) + " " + initial_state.strip() + "\n",
        rwd_fn=rwd_fn,
        v_fn=v_fn,
        alpha=alpha,
        depth=0,
        max_depth=horizon
    )
    tot_sample = 0
    while not cur_node.is_terminal:
        new_node, tmp_sample = planner(cur_node)
        tot_sample += tmp_sample
        cur_node = deepcopy(new_node)
    return cur_node.achieved_goal, tot_sample