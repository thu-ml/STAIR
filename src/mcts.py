from mcts_node import mcts_node
from typing import Optional, Dict, Any, List, Type
import random
import queue

class mcts:
    root: Optional[mcts_node] = None
    tree_size: int = 0 

    def __init__(self, root_node):
        self.root = root_node
        self.tree_size = 1

    # Select node until reach a leaf node
    def select(self, able_to_reselected, node = None, score_type = "UCB"):
        work_node = node if node != None else self.root
        if (not able_to_reselected) and (work_node.is_valid_selected == False):
            return None
        while work_node.children != []:
            best_score = -100
            candidate_children = []
            for child in work_node.children:
                if (not able_to_reselected) and (child.is_valid_selected == False):
                    continue
                score = child.score(score_type)
                if score > best_score:
                    best_score = score
                    candidate_children = [child]
                elif score == best_score:
                    candidate_children.append(child)
            assert candidate_children != [], "is_valid_selected set wrong"
            work_node = random.choice(candidate_children)
        return work_node

    # Expand one node
    def add_node(self, old_node:mcts_node, child_output, child_p):
        new_node = mcts_node(old_node, child_output, child_p)
        old_node.add_child(new_node)
        self.tree_size += 1

    # Provide evaluate feedback and backpropagate
    def update(self, node:mcts_node, feedback, is_terminal, feedback_type):
        work_node = node
        work_node.update(feedback, is_terminal, feedback_type)
        while work_node.parent != None:
            work_node = work_node.parent
            work_node.update(feedback, False, "backpropagation")

    # Return all node information in the tree
    def show_tree(self):
        result = {}
        result["tree_size"] = self.tree_size
        tags = queue.Queue()
        nodes = queue.Queue()
        tags.put("0")
        nodes.put(self.root)
        while not tags.empty():
            tag = tags.get()
            node = nodes.get()
            result[tag] = node.node_info()
            for index in range(len(node.children)):
                tags.put(tag+"."+str(index))
                nodes.put(node.children[index])
        return result
