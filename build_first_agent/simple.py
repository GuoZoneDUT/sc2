import time
import numpy as np

from pysc2.agents import base_agent
from pysc2.lib import actions, features

_NOOP = actions.FUNCTIONS.no_op.id#空闲
_SELECT_POINT = actions.FUNCTIONS.select_point.id#选择单位
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index#待更新
_MOVE_MINIMAP = actions.FUNCTIONS.Move_minimap.id#行走
#单位的代号,定义可见
# https://github.com/Blizzard/s2client-api/blob/master/include/sc2api/sc2_typeenums.h
_TERRAN_SCV = 45
#待更新
_SCREEN = [0]
_MINIMAP= [1]
#定义自己的agent类
class test_agent(base_agent.BaseAgent):
    scv_selected = False
    #定义动作
    def step(self, obs):
        super(test_agent, self).step(obs)
        #判断scv是否选择
        if not self.scv_selected:
            unit_type = obs.observation['screen'][_UNIT_TYPE]
            unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()
            target = [unit_x[0], unit_y[0]]
            self.scv_selected = True
            #选择scv
            return actions.FunctionCall(_SELECT_POINT, [_SCREEN, target])
        else:
            #走到小地图[25,25]处
            return actions.FunctionCall(_MOVE_MINIMAP,[_MINIMAP,[25,25]])
        return actions.FunctionCall(_NOOP, [])