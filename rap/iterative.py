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
sys.path.append("gpt-plan-benchmark/gpt_plan_test")
from utils import *
from tqdm import tqdm


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

    def _expand(self, node: ITERSNode):
        if node not in self.children:
            self.children[node] = node.find_children()

    def _max_ahead(self, paths: list[list[ITERSNode]]):
        return max(paths, key=lambda x: self.M[x[0]])