from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import random

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

from sc_agent.wc_common import *

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_NOT_QUEUED = [0]
_SELECT_ALL = [0]


class MoveToBeacon(base_agent.BaseAgent):
    """An agent specifically for solving the MoveToBeacon map."""

    def step(self, obs):
        super(MoveToBeacon, self).step(obs)
        if _MOVE_SCREEN in obs.observation["available_actions"]:
            player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
            neutral_y, neutral_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
            if not neutral_y.any():
                return actions.FunctionCall(_NO_OP, [])
            target = [int(neutral_x.mean()), int(neutral_y.mean())]
            return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])
        else:
            return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])


class CollectMineralShards(base_agent.BaseAgent):
    """An agent specifically for solving the CollectMineralShards map."""

    def step(self, obs):
        super(CollectMineralShards, self).step(obs)
        if _MOVE_SCREEN in obs.observation["available_actions"]:
            player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
            neutral_y, neutral_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
            player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
            if not neutral_y.any() or not player_y.any():
                return actions.FunctionCall(_NO_OP, [])
            player = [int(player_x.mean()), int(player_y.mean())]
            closest, min_dist = None, None
            for p in zip(neutral_x, neutral_y):
                dist = numpy.linalg.norm(numpy.array(player) - numpy.array(p))
                if not min_dist or dist < min_dist:
                    closest, min_dist = p, dist
            return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, closest])
        else:
            return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])


class DefeatRoaches(base_agent.BaseAgent):
    """An agent specifically for solving the DefeatRoaches map."""

    def step(self, obs):
        super(DefeatRoaches, self).step(obs)
        if _ATTACK_SCREEN in obs.observation["available_actions"]:
            player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
            roach_y, roach_x = (player_relative == _PLAYER_HOSTILE).nonzero()
            if not roach_y.any():
                return actions.FunctionCall(_NO_OP, [])
            index = numpy.argmax(roach_y)
            target = [roach_x[index], roach_y[index]]
            return actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, target])
        else:
            return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])


class GroupAction:
    def __init__(self, name, atom_action_list=list(), init_step_index=0, priority=0):
        self.atom_action_list = atom_action_list
        self.step_index = init_step_index
        self.priority = priority
        self.name = name

    def priority(self):
        return self.priority

    def name(self):
        return self.name

    def finish(self):
        if self.step_index >= len(self.atom_action_list):
            return True
        return False

    def step(self, obs):
        ret = actions.FunctionCall(_NO_OP, [])
        curr_act = self.atom_action_list[self.step_index]
        if curr_act == "select_idle_scv":
            ret = actions.FunctionCall(actions.FUNCTIONS.select_idle_worker.id, [_SELECT_ALL])
            print("interpret_action = " + str(curr_act) + " , " + str([actions.FUNCTIONS.select_idle_worker.id, [_SELECT_ALL]]))
        elif curr_act == "select_one_scv":
            ret = actions.FunctionCall(actions.FUNCTIONS.select_unit.id, [[WC_UNIT_TYPEID["TERRAN_SCV"]], [0]])
            print("interpret_action = " + str(curr_act) + " , " + str(
                [[WC_UNIT_TYPEID["TERRAN_SCV"]], [0]]))
        elif curr_act == "move_to_mineral":
            if _MOVE_SCREEN in obs.observation["available_actions"]:
                player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
                neutral_y, neutral_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
                player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
                if not neutral_y.any() or not player_y.any():
                    return actions.FunctionCall(_NO_OP, [])
                player = [int(player_x.mean()), int(player_y.mean())]
                closest, min_dist = None, None
                for p in zip(neutral_x, neutral_y):
                    dist = numpy.linalg.norm(numpy.array(player) - numpy.array(p))
                    if not min_dist or dist < min_dist:
                        closest, min_dist = p, dist
                ret = actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, closest])
                print("interpret_action = " + str(curr_act) + " , " + str(
                    [_MOVE_SCREEN, [_NOT_QUEUED, closest]]))
        elif curr_act == "move_to_gas":
            if _MOVE_SCREEN in obs.observation["available_actions"]:
                player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
                neutral_y, neutral_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
                player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
                if not neutral_y.any() or not player_y.any():
                    return actions.FunctionCall(_NO_OP, [])
                player = [int(player_x.mean()), int(player_y.mean())]
                closest, min_dist = None, None
                for p in zip(neutral_x, neutral_y):
                    dist = numpy.linalg.norm(numpy.array(player) - numpy.array(p))
                    if not min_dist or dist < min_dist:
                        closest, min_dist = p, dist
                ret = actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, closest])
                print("interpret_action = " + str(curr_act) + " , " + str(
                    [_MOVE_SCREEN, [_NOT_QUEUED, closest]]))
        elif curr_act == "build_one_barrack":
            random_x = random.randint(0, 50)
            random_y = random.randint(0, 50)
            ret = actions.FunctionCall(actions.FUNCTIONS.Build_Barracks_screen.id, [[WC_ABILITY_ID["BUILD_BARRACKS"]], [0], [random_x, random_y]])
            print("interpret_action = " + str(curr_act) + " , " + str(
                [actions.FUNCTIONS.Build_Barracks_screen.id, [[WC_ABILITY_ID["BUILD_BARRACKS"]], [0], [random_x, random_y]]]))
        elif curr_act == "":
            pass
        elif curr_act == "":
            pass
        elif curr_act == "":
            pass

        self.step_index += 1
        return ret


class SimpleTerranAgent(base_agent.BaseAgent):
    """An agent specifically for solving the CollectMineralShards map."""
    def __init__(self):
        base_agent.BaseAgent.__init__(self)
        self.action_model = None
        self.action_interpreter = None
        self.group_action_list = []

    def interpret_action_by_rule(self, obs, next_action):
        print("interpret_action = " + str(next_action))
        group_action = None
        atom_action_list = []
        if next_action == "Collect_Mineral":
            atom_action_list.append(["select_one_scv"])
            atom_action_list.append(["move_to_mineral"])
        elif next_action == "Collect_Gas":
            atom_action_list.append(["select_idle_scv"])
            atom_action_list.append(["move_to_gas"])
        elif next_action == "Build_Barrack":
            atom_action_list.append(["select_one_scv"])
            atom_action_list.append(["build_one_barrack"])
        elif next_action == "Train_Marine":
            atom_action_list.append(["select_one_barrack"])
            atom_action_list.append(["train_one_marine"])
        elif next_action == "Train_SCV":
            atom_action_list.append(["select_one_base"])
            atom_action_list.append(["train_one_scv"])

        if len(atom_action_list) > 0:
            group_action = GroupAction(next_action, atom_action_list, init_step_index=0, priority=0)

        return group_action

    def add_group_action(self, group_action):
        if group_action is not None:
            self.group_action_list.append(group_action)
            if group_action.priority() > 0:
                self.group_action_list = sorted(self.group_action_list, key=lambda x: x.priority(), reverse=True)

    def clear_group_action(self):
        self.group_action_list = []

    def step(self, obs):
        super(SimpleTerranAgent, self).step(obs)

        next_action = self.predict_next_action(obs)
        group_action = self.interpret_action(obs, next_action)

        if len(self.group_action_list) == 0:
            self.add_group_action(group_action)
        return self.do_group_action(obs)

    def predict_next_action(self, obs):
        if self.action_model is not None:
            next_action = self.action_model.predict(obs)
        else:
            next_action = self.predict_next_action_by_rule(obs)
        return next_action

    def interpret_action(self, obs, next_action):
        if self.action_interpreter is not None:
            action_info = self.action_interpreter.explain(obs, next_action)
        else:
            action_info = self.interpret_action_by_rule(obs, next_action)
        return action_info

    def predict_next_action_by_rule(self, obs):
        # action_list = ["Collect_Mineral", "Collect_Gas", "Build_Barrack", "Train_Marine", "Train_SCV"]
        action_list = ["Collect_Mineral", "Collect_Gas", "Build_Barrack"]
        select_action = random.choice(action_list)
        return select_action

    def do_group_action(self, obs):
        if len(self.group_action_list) == 0:
            return actions.FunctionCall(_NO_OP, [])
        group_action = self.group_action_list[0]
        ret = group_action.step(obs)
        if group_action.finish():
            self.group_action_list.pop(0)
        return ret




