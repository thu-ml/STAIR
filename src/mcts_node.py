from typing import Optional, Dict, Any, List, Type
from math import sqrt, log
import random

class mcts_node:
    parent: Optional[Any] = None
    children: Optional[List[Any]] = None
    prompt: str = ""
    trajectory: List[str] = [] # Partial solution so far
    is_terminal: bool = False
    is_valid_selected: bool = True # Whether can be selected in next turn if "able_to_reselected" set to False
    depth: int = 0
    max_depth: int = 10
    visited_times: int = 0
    value_sum: int = 0 # value_sum / visited_times -> "true" value
    reward: Optional[int] = None # Reward provided by ORM for terminal node
    c: float = 1.5 # Hyperparam for PUCB / UCB
    p: float = 1.0 # "Probability" of the action

    # Initialize
    def __init__(self, parent, output, p):
        self.parent = parent
        if parent != None:
            self.prompt = parent.prompt
            self.trajectory = parent.trajectory + [output]
            self.depth = parent.depth + 1
            self.max_depth = parent.max_depth
            self.c = parent.c
        self.p = p
        self.children = []

    # Add child
    def add_child(self, child):
        self.children.append(child)

    # Update node information -> a visit
    def update(self, feedback, is_terminal, feedback_type):
        if feedback != -100: # feedback == -100 if not actually visit, but only to refresh the tree state. And it indicates that the node cannot expand more node.
            if self.visited_times == 0: # First time visit
                self.is_terminal = is_terminal
                if feedback_type == "reward":
                    self.reward = feedback
            self.value_sum += feedback
            self.visited_times += 1
        self.is_valid_selected = False
        if self.is_terminal == False and self.depth < self.max_depth and self.children == [] and feedback != -100:
            # If is a leaf node and can expand more node
            self.is_valid_selected = True
        else:
            # If is an inner node and has a child that valid to be selected
            for child in self.children:
                if child.is_valid_selected:
                    self.is_valid_selected = True
                    break

    # Return node score for required score type
    def score(self, score_type):
        if score_type == "PUCB":
            return self.PUCB_score()
        if score_type == "UCB":
            return self.UCB_score()
        assert False, "Wrong score type"

    # Return node score in selection phase (PUCB)
    def PUCB_score(self):
        if self.visited_times == 0:
            return 100 * self.c * self.p
        v_score = self.value()
        u_score = self.c * self.p * sqrt(self.parent.visited_times) / (1 + self.visited_times)
        return v_score + u_score

    # Return node score in selection phase (UCB)
    def UCB_score(self):
        if self.visited_times == 0:
            return 100 * self.c 
        u_score = self.c * sqrt(log(self.parent.visited_times) / self.visited_times)
        v_score = self.value()
        return v_score + u_score

    # Return node value
    def value(self):
        assert self.visited_times != 0, "Node has not been visited yet but require for value"
        return self.value_sum / self.visited_times

    # Return dict about node information
    def node_info(self):
        info = {}
        info["prompt"] = self.prompt
        info["trajectory"] = self.trajectory
        info["is_terminal"] = self.is_terminal
        info["is_valid_selected"] = self.is_valid_selected
        info["depth"] = self.depth
        info["visited_times"] = self.visited_times
        info["reward"] = self.reward
        info["value_sum"] = self.value_sum
        info["true_value"] = self.value() if self.visited_times > 0 else None
        info["p"] = self.p
        return info

    # Set param (only needed for root node)
    def set_param(self, c, max_depth, question, depth, trajectory):
        self.c = c
        self.max_depth = max_depth
        self.prompt = question
        self.depth = depth
        self.trajectory = trajectory
