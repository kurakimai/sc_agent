from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

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


class SimpleTerranAgent(base_agent.BaseAgent):
    """An agent specifically for solving the CollectMineralShards map."""
    def __init__(self):
        base_agent.BaseAgent.__init__(self)
        self.action_model = None
        self.action_interpreter = None
        self.atom_action_list = []

    def add_atom_action(self, action_id, arg_list, priority=0):
        self.atom_action_list.append([action_id, arg_list, priority])

    def clear_atom_action(self):
        self.atom_action_list = []

    def do_action(self, action_info):
        if action_info is not None:
            pass

        if len(self.atom_action_list) > 0:
            c_act = self.atom_action_list.pop(0)
            return actions.FunctionCall(c_act[0], c_act[1])
        return actions.FunctionCall(_NO_OP, [])

    def step(self, obs):
        super(SimpleTerranAgent, self).step(obs)
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


    def predict_next_action(self, obs):
        if self.action_model is not None:
            next_action = self.action_model.predict(obs)
        else:
            next_action = self.predict_next_action_by_rule(obs)

        if self.action_interpreter is not None:
            action_info = self.action_interpreter.explain(obs, next_action)
        else:
            action_info = self.explain_action_by_rule(obs, next_action)

        return self.do_action(action_info)


    def predict_next_action_by_rule(self, obs):
        return ""


    def explain_action_by_rule(self, obs, next_action):
        return {}



