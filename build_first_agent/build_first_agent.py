#This code can't work.I will update soon

import time
import numpy as np

from pysc2.agents import base_agent
from pysc2.lib import actions,features

#行为代码
_Train_Drone_quick = actions.FUNCTIONS.Train_Drone_quick.id#训练幼虫
_BUILD_SPAWNINGPOOL_SCREEN = actions.FUNCTIONS.Build_SpawningPool_screen.id#变形血池
_RALLY_UNITS_SCREEN = actions.FUNCTIONS.Rally_Units_screen.id#设置集结点

_NOOP = actions.FUNCTIONS.no_op.id
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_SELECT_POINT = actions.FUNCTIONS.select_point.id
#单位代码
_ZERG_DRONE = 104
_ZERG_LARVA = 151 
_ZERG_HATCHERY = 86
_ZERG_ZERGING = 105
_SCREEN = [0]
class FirstAgent(base_agent.BaseAgent):

    def transformLocation(self,x,x_distance,y,y_distance):
        return[x - x_distance,y - y_distance]
    def step(self,obs):
        super(FirstAgent,self).step(obs)

        unit_type = obs.observation['screen'][_UNIT_TYPE]
        
        unit_y,unit_x = (unit_type == _ZERG_DRONE).nonzero()

        target = [unit_x[0],unit_y[0]]
        
        actions.FunctionCall(_SELECT_POINT,[_SCREEN,target])
        if _BUILD_SPAWNINGPOOL_SCREEN in obs.observation['available_actions']:
            unit_type = obs.observation['screen'][_UNIT_TYPE]

            unit_y,unit_x = (unit_type == _ZERG_HATCHERY).nonzero()

            target = self.transformLocation(int(unit_x.mean()),10,int(unit_y.mean()),0)
            actions.FunctionCall(_Build_SpawningPool_screen,[_SCREEN,target])

        return actions.FunctionCall(_NOOP,[])
        
