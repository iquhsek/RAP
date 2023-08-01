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
        return 0

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
        for k in range(max_iter):
            # print(f'{Fore.MAGENTA} Rollout # {k} --------------{Style.RESET_ALL}')
            # generate candidate lookahead paths
            paths = self._lookahead(node)
            # calculate the return from each path
            # print(f'{Fore.MAGENTA}Back propagate paths one by one...{Style.RESET_ALL}') # TODO: debug
            for path in paths:
                self._back_propagate(path)
            # choose the path with maximum return
            max_path = self._max_ahead(paths)
            # TODO: debug. The GOAL is the same for all nodes so we don't have to print prompt["GOAL"]
            # print(f'{Fore.MAGENTA}Now print the max path we chose in this step in the form (prompt, reward) node by node --> {Style.RESET_ALL}')
            # for tmp_node in max_path:
                # print('--------go node--------')
                # print(tmp_node.prompt)
                # print(f'reward={tmp_node.reward},r0={tmp_node._r0},r1={tmp_node._r1},max_return={self.M[tmp_node]}')
                # print('-----------------------')
            # the next node is the ending node of the chosen path
            next_node = max_path[-1]
            # stop iteration if we reached the goal
            if next_node.achieved_goal:
                break
        # print(f'{Fore.BLUE}--------------Rollout END--------------{Style.RESET_ALL}')
        return next_node

    def _back_propagate(self, path: list[ITERSNode], reward=0.):
        coeff = 1
        # print(f'{Fore.RED}Start back-propagating a path in a reverse order{Style.RESET_ALL}')
        for node in reversed(path):
            # print(f'{Fore.RED}The current has prompt={Style.RESET_ALL}{node.prompt}')
            reward = reward * self.discount + node.reward
            # print(f'{Fore.RED}Counted its return={Style.RESET_ALL}{reward}')
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
        # print(f'{Fore.MAGENTA}Comparing M values of path lens {[len(x) for x in paths]}{Style.RESET_ALL}')
        # print(f'{Fore.MAGENTA}Comparing M values in {[self.M[x[0]] for x in paths]}{Style.RESET_ALL}')
        return max(paths, key=lambda x: self.M[x[0]])


class PITERS(ITERS):
    def __init__(self, w_exp=1, discount=1, prior=False, aggr_reward='sum', aggr_child='max', sample_per_node=2):
        super().__init__(w_exp, discount, prior, aggr_reward, aggr_child)
        self.sample_per_node = sample_per_node

    def _lookahead(self, node: ITERSNode):
        paths = []
        def route(node, path):
            self._expand(node)
            if node.is_terminal:
                paths.append(path)
            else:
                # print(f'Inside lookahead method, the current node has {len(self.children[node])} many children')
                self.children[node].sort(reverse=True, key=lambda x: x._r0)
                children_sample = self.children[node][:self.sample_per_node] if self.sample_per_node != 0 else self.children[node]
                # print(f'Inside lookahead method, we sampled {len(children_sample)} many children')
                for new_node in children_sample:
                    tmp_path = deepcopy(path)
                    tmp_path.append(new_node)
                    route(new_node, tmp_path)
        route(node, [])
        return paths
