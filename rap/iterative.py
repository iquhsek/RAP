import io
import json
import os
import math
import random
import re
import sys
import traceback
import warnings
from copy import deepcopy
from collections import defaultdict
from abc import ABC, abstractmethod
sys.path.append("gpt-plan-benchmark/gpt_plan_test")
from utils import *
from tqdm import tqdm

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
        raise NotImplementedError
        # return 0

    @property
    @abstractmethod
    def visited(self):
        return 0


class ITERS:
    """Iterative lookahead planning with llm"""
    def __init__(self, w_exp=1, discount=1, prior=False, aggr_reward='sum', aggr_child='max'):
        self.Q: dict[ITERSNode, float] = defaultdict(lambda : 0.)
        self.N: dict[ITERSNode, int] = defaultdict(lambda : 0)
        self.M: dict[ITERSNode, float] = defaultdict(lambda : -math.inf)
        self.children = dict()
        self.w_exp = w_exp
        self.discount = discount
        self.prior = prior
        self.aggr_reward = aggr_reward
        self.aggr_child = aggr_child

    def rollout(self, max_iter: int, node: ITERSNode):
        # TODO: debug
        # print(f'{Fore.BLUE}--------------BEGIN a rollout with {max_iter} loops--------------{Style.RESET_ALL}')
        cur_node = deepcopy(node)
        for k in range(max_iter):
            # generate candidate lookahead paths
            paths = self._lookahead(cur_node)
            # calculate the return from each path
            for path in paths:
                self._back_propagate(path)
            # choose the path with maximum return
            if all(path[-1]._r1 == paths[0][-1]._r1 for path in paths):
                max_path = self._rand_ahead(paths)
            else:
                max_path = self._max_ahead(paths)
            next_node = max_path[-1]
            print(f'depth of next node={next_node.depth}')
            # stop iteration if we reached the goal
            if next_node.achieved_goal:
                print('YES ACHIEVED')
                break
            if next_node.is_terminal:
                print('EXCEEDED MAX_DEPTH')
                break
            cur_node = deepcopy(next_node)
        # print(f'{Fore.BLUE}--------------Rollout END--------------{Style.RESET_ALL}')
        return next_node

    def _back_propagate(self, path: list[ITERSNode], reward=0.):
        coeff = 1
        # print(f'{Fore.RED}Start back-propagating a path in a reverse order{Style.RESET_ALL}', end='|||||')
        for node in reversed(path):
            # print(f'{Fore.RED}The current has prompt={Style.RESET_ALL}{node.prompt}', end='|||||')
            reward = reward * self.discount + node._r0 * node._r_alpha
            reward += node._r1 if node == path[-1] else 0 # add Vrand if this node is at the end of the forward search
            # print(f'{Fore.RED}Counted its return={Style.RESET_ALL}{reward}', end='|||||')
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
        # print(f'{Fore.RED}End this back-propagating{Style.RESET_ALL}')
        # print()

    def _lookahead(self, node: ITERSNode):
        # print(f'{Fore.MAGENTA}----------------look ahead BEGIN----------------{Style.RESET_ALL}') # TODO: debug
        # print(f'{Fore.MAGENTA}Recursively generate paths...{Style.RESET_ALL}') # TODO: debug
        paths = []
        def route(node, path):
            self._expand(node)
            if node.is_terminal:
                paths.append(path)
            else:
                for new_node in self.children[node]:
                    tmp_path = deepcopy(path)
                    tmp_path.append(new_node)
                    route(new_node, tmp_path)
        route(node, [])
        # print(f'{Fore.MAGENTA}Got in total {len(paths)} paths.{Style.RESET_ALL}') # TODO: debug
        # print(f'{Fore.MAGENTA}----------------look ahead END----------------{Style.RESET_ALL}') # TODO: debug
        return paths

    def _expand(self, node: ITERSNode):
        if node not in self.children:
            self.children[node] = node.find_children()

    def _max_ahead(self, paths: list[list[ITERSNode]]):
        return max(paths, key=lambda x: self.M[x[0]])

    def _rand_ahead(self, paths: list[list[ITERSNode]]):
        return random.choices(paths, weights=[self.M[path[0]] for path in paths])[0]


class PITERS(ITERS):
    def __init__(self, w_exp=1, discount=1, prior=False, aggr_reward='sum', aggr_child='max', sample_per_node=2, search_depth=6):
        super().__init__(w_exp, discount, prior, aggr_reward, aggr_child)
        self.sample_per_node = sample_per_node
        self.search_depth = search_depth

    def _lookahead(self, father_node: ITERSNode):
        paths = []
        def route(node, path):
            self._expand(node)
            if node.depth - father_node.depth >= self.search_depth or node.is_terminal:
                paths.append(path)
            else:
                self.children[node].sort(reverse=True, key=lambda x: x._r0)
                children_sample = self.children[node][:self.sample_per_node] if self.sample_per_node != 0 else self.children[node]
                for new_node in children_sample:
                    tmp_path = deepcopy(path)
                    tmp_path.append(new_node)
                    route(new_node, tmp_path)
        route(father_node, [])
        return paths
