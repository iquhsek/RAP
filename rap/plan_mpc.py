import re
import json
import torch
import numpy as np
from copy import deepcopy
from functools import partial
from typing import List, Tuple
from rap.models import QueryLM
from rap.plan_reflex import MemStateNode
from rap.forward_search import ForwardSearch
from rap.utils.blocksworld import apply_change, generate_all_actions, get_world_change

with open('data/blocksworld/critique_prompt.txt', 'r') as f:
    REFLEX_PREFIX = f.read()

with open('data/blocksworld/direct_value_prompt.txt', 'r') as f:
    VRAND_PREFIX = f.read()

with open('data/blocksworld/my_mcts_prompts_update.json', 'r') as f:
    SPELLS = json.load(f)


class MPCNode(MemStateNode):
    def __init__(self, prompt, memory, rwd_fn, v_fn, alpha, depth, max_depth, parent=None, prob_r=0):
        self.children = []
        self.depth = depth # real depth, not relative depth
        self.max_depth = max_depth
        self.parent = parent
        self._alpha = alpha
        self.rwd_fn = rwd_fn
        self.v_fn = v_fn
        self._prob_r = prob_r
        self.prompt = v_fn(prompt, depth)


def _get_v_rand(critic: QueryLM, node: MPCNode) -> int:
    goal_statement = node.prompt.split("[GOAL]")[-1].split("[STATE 0]")[0]
    last_state = re.search(f'.*{re.escape(SPELLS["state_prefix"].format(node.depth - 1))}(.*)', node.prompt)[1]
    prompt = VRAND_PREFIX.format(goal_statement, last_state.rstrip('\n'))
    response = critic.query_LM(prompt).rstrip('\n')
    return int(response)


def v_fn(world_model: QueryLM, inp, depth) -> str:
    if depth == 0:
        return inp
    last_state = re.search(f'.*{re.escape(SPELLS["state_prefix"].format(depth - 1))}(.*)', inp)[1]
    last_action = re.search(f'.*{re.escape(SPELLS["action_prefix"].format(depth))}(.*)', inp)[1]
    world_change = get_world_change(last_state, last_action)
    last_state = inp.split(f"[STATE {depth-1}]")[-1].split(f"[ACTION {depth}]")[0]
    new_state = apply_change(world_change, last_state)
    new_prompt = inp + SPELLS["state_prefix"].format(depth) + " " + new_state + "\n"
    return new_prompt


def rwd_fn(estimator: QueryLM, inp: str, memory: str, depth: int) -> Tuple[List, np.ndarray]:
    '''For r=Pr(a|s_t), the probability component reward'''
    last_state = re.search(f'.*{re.escape(SPELLS["state_prefix"].format(depth))}(.*)', inp)[1]
    raw_action_list = generate_all_actions(last_state)
    action_output = [inp + SPELLS["action_prefix"].format(depth + 1) + " " + a.capitalize() + ".\n" for a in raw_action_list]
    n_base_actions = 2 * (depth // 2)
    last_base_state = inp.split(SPELLS["state_prefix"].format(n_base_actions))[-1].split(SPELLS["action_prefix"].format(n_base_actions + 1))[0].strip()
    # Task description
    baseline_prompt = SPELLS["basebaseline_action1"]
    # Add memory
    if memory is not None:
        assert type(memory) is str
        baseline_prompt += "\nYour memory for the task below:\n"
        baseline_prompt += memory
    # Add baseline statements
    baseline_prompt += "\n\n"
    baseline_prompt += SPELLS["basebaseline_action2"]
    # Add current new statement
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
        if estimator.__class__.__name__ == 'QueryVicuna':
            log_probs = estimator.get_ll(baseline_prompt, ll_prompts[idx: end_idx])
        else:
            log_probs = estimator.llamamodel.get_ll(baseline_prompt, ll_prompts[idx: end_idx])
        scores += list(log_probs)
    scores = np.array(scores, dtype=np.dtype('float64'))
    exp_scores = np.exp(scores)
    soft_scores = exp_scores / np.sum(exp_scores)
    return new_action_output, soft_scores


class ValueIteration(ForwardSearch):
    def __call__(self, father_node: MPCNode) -> Tuple[MPCNode, int]:
        paths = []
        returns = []
        def route(node: MPCNode, path: List[MPCNode]) -> None:
            if node.depth - father_node.depth >= self.search_depth or node.is_terminal:
                paths.append(path)
                # compute cumulative rewards
                c_rwd = 0
                for s in path:
                    c_rwd = c_rwd * self.discount + s._prob_r
                c_rwd += _get_v_rand(node) * node._alpha # Add v_rand at the end
                assert c_rwd > 0
                returns.append(c_rwd)
            else:
                children_sample = self.sampler(node)
                for new_node in children_sample:
                    tmp_path = deepcopy(path)
                    tmp_path.append(new_node)
                    route(new_node, tmp_path)
        # recursively generate feasible plans
        route(father_node, [])
        # find the approximately best plan
        max_id = np.argmax(returns)
        # only take one actual step
        next_node = paths[max_id][0]
        assert next_node.depth == father_node.depth + 1
        # count samples
        tmp_sample = np.sum(len(path) for path in paths)
        return next_node, tmp_sample


def mpc_plan(initial_state: str,
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

    global _get_v_rand, rwd_fn, v_fn
    _get_v_rand = partial(_get_v_rand, critic=world_model)
    rwd_fn = partial(rwd_fn, estimator=world_model)
    v_fn = partial(v_fn, world_model=world_model)

    memory=None
    planner = ValueIteration(
        search_depth=search_depth,
        sample_per_node=sample_per_node,
        sampler=sampler,
        discount=discount
    )
    cur_node = MPCNode(
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
        reflexion_prompt = REFLEX_PREFIX.format(cur_node.prompt.rstrip('\n'))
        reflexion_output = world_model.query_LM(
            prompt=reflexion_prompt,
            do_sample=False,
        )
        memory = reflexion_prompt.split("[GOAL]")[-1] + reflexion_output
        # Renew and restart the node
        cur_node = MPCNode(
            prompt=prompts["goal_prefix"] + goal.strip() + "\n" + prompts["state_prefix"].format(0) + " " + initial_state.strip() + "\n",
            memory=memory,
            rwd_fn=rwd_fn,
            v_fn=v_fn,
            alpha=alpha,
            depth=0,
            max_depth=horizon
        )
    return False, tot_sample