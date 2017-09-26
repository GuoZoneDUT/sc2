import time
import numpy as np

from pysc2.agents import base_agent
from pysc2.lib import actions, features

# 定义动作
_NOOP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_BUILD_SUPPLYDEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_FACTORY = actions.FUNCTIONS.Build_Factory_screen.id
_BUILD_STARPORT = actions.FUNCTIONS.Build_Starport_screen.id
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index

# 定义单位数字
_TERRAN_FACTORY = 27
_TERRAN_SUPPLYDEPOT = 19
_TERRAN_SCV = 45
_TERRAN_BARRACKS = 21
_TERRAN_COMMANDCENTER = 18
_TERRAN_STARPORT = 28

_PLAYER_SELF = 1
_SCREEN = [0]


class build_init_agent(base_agent.BaseAgent):
    #状态初始化
    agent_init={'scv_selected':False,
                'supplydepot_built':False,
                'barracks_built':False,
                'base_top_left':False}
    #初始化情节数
    episodes_count=1;
    # 根据所处位置改变坐标转换操作
    def transformLocation(self, x, x_distacne, y, y_distance):
        if not self.agent_init['base_top_left']:
            return [x - x_distacne, y - y_distance]
        return [x + x_distacne, y + y_distance]

    def step(self, obs):
        super(build_init_agent, self).step(obs)
        #如果进入到下一次情节,重新初始化agent变量
        if self.episodes != self.episodes_count:
            for i in self.agent_init:
                self.agent_init[i]=False
                self.episodes_count=self.episodes

        # 是否位于左上角
        if self.agent_init['base_top_left'] is False:
            player_y, player_x = (obs.observation["minimap"][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
            self.agent_init['base_top_left'] = player_y.mean() <= 31

        # 是否选中SCV
        if not self.agent_init['scv_selected']:
            unit_type = obs.observation['screen'][_UNIT_TYPE]
            unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()
            target = [unit_x[0], unit_y[0]]
            self.agent_init['scv_selected'] = True
            # 选中SCV
            return actions.FunctionCall(_SELECT_POINT, [_SCREEN, target])
        else:
            # 是否已经建造补给站
            if not self.agent_init['supplydepot_built']:
                if _BUILD_SUPPLYDEPOT in obs.observation['available_actions']:
                    unit_type = obs.observation['screen'][_UNIT_TYPE]
                    unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
                    target = self.transformLocation(int(unit_x.mean()), 0, int(unit_y.mean()), 20)
                    self.agent_init['supplydepot_built'] = True
                    # 建造补给站
                    return actions.FunctionCall(_BUILD_SUPPLYDEPOT, [_SCREEN, target])
            else:
                # 是否建造兵营
                if not self.agent_init['barracks_built']:
                    if _BUILD_BARRACKS in obs.observation['available_actions']:
                        unit_type = obs.observation['screen'][_UNIT_TYPE]
                        unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
                        target = self.transformLocation(int(unit_x.mean()), 20, int(unit_y.mean()), 0)
                        self.agent_init['barracks_built'] = True
                        # 建造兵营
                        return actions.FunctionCall(_BUILD_BARRACKS, [_SCREEN, target])
        return actions.FunctionCall(_NOOP, [])



