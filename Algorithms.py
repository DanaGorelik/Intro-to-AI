import numpy as np
#from DragonBallEnv import DragonBallEnv
from typing import List, Tuple
import heapdict
from collections import OrderedDict

class Node():
    def __init__(self, state, node) -> None:
        self.state = state
        self.parent = node
        self.actions = []
        self.total_cost = 0
        self.terminated = False

    def update_node(self, cost, terminated, last_action):
        self.actions = self.parent.actions.copy()
        self.actions.append(last_action)
        self.total_cost = self.parent.total_cost + cost
        self.terminated = terminated

class BFSAgent():
    def solution(self, node, expanded):
        return (node.actions, node.total_cost, expanded)

    def search(self, env: DragonBallEnv) -> Tuple[List[int], float, int]:
        initial_Node = Node(env.get_initial_state(), None)
        if env.is_final_state(env.get_initial_state()):
          return self.solution(initial_Node, 0)
        countExpanded = 0
        OpenDict = OrderedDict()
        OpenDict[initial_Node.state] = initial_Node
        close = set()
        while OpenDict:
          currNode = OpenDict.popitem(last = False)[1]
          close.add(currNode.state)
          env.reset()
          env.set_state(currNode.state)
          countExpanded += 1
          for goal in env.get_goal_states():
            if currNode.state[0] == goal[0] and (currNode.state[1] == False or currNode.state[2] == False):
              continue
          dictOfSucc = env.succ(currNode.state)
          for a, (state,cost,terminated) in dictOfSucc.items():
            if state == None:
              break
            new_state, cost,terminated = env.step(a)
            childNode = Node(new_state, currNode)
            childNode.update_node(cost, terminated, a)
            if childNode.state not in close and childNode.state not in OpenDict:
              if env.is_final_state(childNode.state):
                return self.solution(childNode, countExpanded)
              else:
                OpenDict[childNode.state] = childNode
            env.reset()
            env.set_state(currNode.state)
        return None  # No path found

class WeightedAStarAgent():

    def solution(node, expanded):
      return (node.actions, node.total_cost, expanded)

    def hManhatten(env, start, end):
      xs, ys = env.to_row_col(start)
      xe, ye = env.to_row_col(end)
      return (abs(xs-xe)+abs(ys-ye))

    def hmsap(env, state):
      goals = env.get_goal_states()
      dist_list = [WeightedAStarAgent.hManhatten(env, state, g) for g in goals]
      if not state[1]:
        dist_list.append(AStarEpsilonAgent.hManhatten(env, state, env.d1))
      if not state[2]:
        dist_list.append(AStarEpsilonAgent.hManhatten(env, state, env.d2))
      return min(dist_list)

    def wa_star(env, h_weight, node):
      return (h_weight*WeightedAStarAgent.hmsap(env, node.state) + (1-h_weight)*node.total_cost)

    def search(self, env: DragonBallEnv, h_weight) -> Tuple[List[int], float, int]:
      #total_cost, Expanded, Actions
      open_heapDict = heapdict.heapdict()
      node = Node(env.get_initial_state(), None)
      open_heapDict[node.state] = (WeightedAStarAgent.wa_star(env, h_weight, node), node.state, node)
      close = {}
      num_expanded = 0
      while len(open_heapDict):
        node_f, node_state, node = open_heapDict.popitem()[1]
        env.reset()
        env.set_state(node.state)
        close[node.state] = (node_f, node)
        if env.is_final_state(node.state):
          return WeightedAStarAgent.solution(node, num_expanded)
        num_expanded += 1 #if num_expanded is here than it's 76
        if not node.terminated:
          for action, (new_state, cost, terminated) in env.succ(node.state).items():
            new_state, cost, terminated = env.step(action)
            child = Node(new_state, node)
            child.update_node(cost, terminated, action)
            f_new = WeightedAStarAgent.wa_star(env, h_weight, child)
            if (child.state not in close) and (child.state not in open_heapDict):
              open_heapDict[child.state] = (f_new, child.state, child)
            elif child.state in open_heapDict:
              f_curr = open_heapDict[child.state][0]
              if f_new < f_curr:
                open_heapDict[child.state] = (f_new, child.state, child)
            else:
              f_curr, n_curr = close[child.state]
              if f_new < f_curr:
                open_heapDict[child.state] = (f_new, child.state, child)
                del close[child.state]
            env.reset()
            env.set_state(node.state)
      return None

class AStarEpsilonAgent():

    def solution(node, expanded):
      return (node.actions, node.total_cost, expanded)

    def hManhatten(env, start, end):
      xs, ys = env.to_row_col(start)
      xe, ye = env.to_row_col(end)
      return (abs(xs-xe)+abs(ys-ye))

    def hmsap(env, state):
      goals = env.get_goal_states()
      dist_list = [AStarEpsilonAgent.hManhatten(env, state, g) for g in goals]
      if not state[1]:
        dist_list.append(AStarEpsilonAgent.hManhatten(env, state, env.d1))
      if not state[2]:
        dist_list.append(AStarEpsilonAgent.hManhatten(env, state, env.d2))
      return min(dist_list)

    def compute_min_f(env,open):
      real_min = [open.peekitem()[1],AStarEpsilonAgent.hmsap(env, open.peekitem()[0]) +open.peekitem()[1][1].total_cost]
      for state, (s, node, node_f) in open.items():
        if real_min[1] > AStarEpsilonAgent.hmsap(env, s) +node.total_cost:
          real_min = [(s,node,node_f),AStarEpsilonAgent.hmsap(env, s) +node.total_cost]
      return real_min

    def next_node_to_expand(env, open, epsilon):
      min_state, min_value = open.peekitem()
      real_min = AStarEpsilonAgent.hmsap(env, min_value[0]) + min_value[1].total_cost
      min_f = AStarEpsilonAgent.compute_min_f(env, open)[1]
      focal = heapdict.heapdict((state, node.total_cost) for state, (s, node, node_f) in open.items() if (AStarEpsilonAgent.hmsap(env, state) + node.total_cost) <= min_f*(1+epsilon))
      return (focal.peekitem()[0],AStarEpsilonAgent.hmsap(env, focal.peekitem()[0]) + focal.peekitem()[1])

    def search(self, env: DragonBallEnv, epsilon: int) -> Tuple[List[int], float, int]:
      open_heapDict = heapdict.heapdict()
      start_node = Node(env.get_initial_state(), None)
      open_heapDict[start_node.state] = (start_node.state, start_node, AStarEpsilonAgent.hmsap(env,start_node.state))
      close = {}
      num_expanded = 0
      while len(open_heapDict):
        state, node_f  = AStarEpsilonAgent.next_node_to_expand(env, open_heapDict,epsilon)
        node = open_heapDict[state][1]
        open_heapDict.pop(node.state)
        env.reset()
        env.set_state(node.state)
        close[node.state] = (node_f, node)
        if env.is_final_state(node.state):
          return AStarEpsilonAgent.solution(node, num_expanded)
        num_expanded += 1
        if not node.terminated:
          for action, (new_state, cost, terminated) in env.succ(node.state).items():
            new_state, cost, terminated = env.step(action)
            child = Node(new_state, node)
            child.update_node(cost, terminated, action)
            f_new = AStarEpsilonAgent.hmsap(env, child.state) + child.total_cost
            if (child.state not in close) and (child.state not in open_heapDict):
              open_heapDict[child.state] = (child.state, child, f_new)
            elif child.state in open_heapDict:
              f_curr = open_heapDict[child.state][2]
              if f_new < f_curr:
                open_heapDict[child.state] = (child.state, child, f_new)
            else:
              f_curr, n_curr = close[child.state]
              if f_new < f_curr:
                open_heapDict[child.state] = (child.state, child, f_new)
                del close[child.state]
            env.reset()
            env.set_state(node.state)
      return None