import re
import torch
import numpy as np
from copy import deepcopy
from typing import List, Tuple
from rap.models import QueryLM
from rap.plan_forward import StateNode
from rap.forward_search import ForwardSearch
from rap.utils.blocksworld import apply_change, generate_all_actions, get_world_change


with open('data/blocksworld/critique_prompt.txt', 'r') as f:
    REFLEX_PREFIX = f.read()


class MemStateNode(StateNode):
    def __init__(self, prompt, memory, rwd_fn, v_fn, alpha, depth, max_depth, parent=None, prob_r=0):
        super().__init__(prompt, rwd_fn, v_fn, alpha, depth, max_depth, parent, prob_r)
        self.memory = memory

    def _create_child_node(self, prompt, prob_r):
        return MemStateNode(prompt, self.memory, self.rwd_fn, self.v_fn, self._alpha, self.depth + 1, self.max_depth, parent=self, prob_r=prob_r)

    def _get_children(self):
        if self.is_terminal:
            return self.children
        questions, arr_prob_r = self.rwd_fn(self.prompt, self.memory, self.depth)
        for question, prob_r in zip(questions, arr_prob_r):
            child = self._create_child_node(question, prob_r)
            if child.achieved_goal:
                return [child]
            self.children.append(child)
        return self.children


def reflex_plan(initial_state: str,
                goal: str,
                prompts: dict,
                world_model: QueryLM,
                alpha: float,
                num_trial: int,
                horizon: int,
                search_depth: int,
                sample_per_node: int,
                sampler: str='heuristic',
                discount: float=1,
                speedup_action_batch_size=2,) -> bool:

    '''-----------------------------------------------------'''

    def rwd_fn(inp: str, memory: str, depth: int) -> Tuple[List, np.ndarray]:
        '''For r=Pr(a|s_t), the probability component reward'''
        last_state = re.search(f'.*{re.escape(prompts["state_prefix"].format(depth))}(.*)', inp)[1]
        raw_action_list = generate_all_actions(last_state)
        action_output = [inp + prompts["action_prefix"].format(depth + 1) + " " + a.capitalize() + ".\n" for a in raw_action_list]
        n_base_actions = 2 * (depth // 2)
        last_base_state = inp.split(prompts["state_prefix"].format(n_base_actions))[-1].split(prompts["action_prefix"].format(n_base_actions + 1))[0].strip()
        # Task description
        baseline_prompt = prompts["basebaseline_action1"]
        # Add memory
        if memory is not None:
            assert type(memory) is str
            baseline_prompt += "\nYour memory for the task below:\n"
            baseline_prompt += memory
        # Add baseline statements
        baseline_prompt += "\n\n"
        baseline_prompt += prompts["basebaseline_action2"]
        # Add current new statement
        baseline_prompt += "\n[STATEMENT]\n"
        baseline_prompt += "As initial conditions " + last_base_state.strip() + "\n" + inp.split("[GOAL]")[-1].split("[STATE 0]")[0].strip() + "\n\nMy plan is as follows:\n\n[PLAN]\n"

        # TODO:
        print()
        print()
        print('-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@')
        print(baseline_prompt)
        print('-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@')
        print()
        print()
        # TODO:

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
        scores = np.array(scores, dtype=np.dtype('float64'))
        exp_scores = np.exp(scores)
        soft_scores = exp_scores / np.sum(exp_scores)
        # TODO:
        print(scores)
        print(exp_scores)
        print(np.sum(exp_scores))
        if memory is not None: print(soft_scores)
        # TODO:
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

    memory=None
    planner = ForwardSearch(
        search_depth=search_depth,
        sample_per_node=sample_per_node,
        sampler=sampler,
        discount=discount
    )
    cur_node = MemStateNode(
        prompt=prompts["goal_prefix"] + goal.strip() + "\n" + prompts["state_prefix"].format(0) + " " + initial_state.strip() + "\n",
        memory=memory,
        rwd_fn=rwd_fn,
        v_fn=v_fn,
        alpha=alpha,
        depth=0,
        max_depth=horizon
    )
    tot_sample = 0
    for _ in range(num_trial):
        while not cur_node.is_terminal:
            new_node, tmp_sample = planner(cur_node)
            tot_sample += tmp_sample
            cur_node = deepcopy(new_node)
        if cur_node.achieved_goal:
            return True, tot_sample
        # Keep the failure as a memory
        reflexion_prompt = REFLEX_PREFIX.format(cur_node.prompt)
        # TODO:
        print()
        print()
        print('reflexion_prompt_last_part-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@')
        print(reflexion_prompt.split("[CRITIC]")[-1])
        print('-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@')
        print()
        print()
        # TODO:
        reflexion_output = world_model.query_LM(
            prompt=reflexion_prompt,
            do_sample=False,
        )
        memory = reflexion_prompt.split("[GOAL]")[-1] + reflexion_output
        # Renew and restart the node
        cur_node = MemStateNode(
            prompt=prompts["goal_prefix"] + goal.strip() + "\n" + prompts["state_prefix"].format(0) + " " + initial_state.strip() + "\n",
            memory=memory,
            rwd_fn=rwd_fn,
            v_fn=v_fn,
            alpha=alpha,
            depth=0,
            max_depth=horizon
        )
    return False, tot_sample