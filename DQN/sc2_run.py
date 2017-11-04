import numpy as np

from pysc2.lib import actions as sc2_action
from pysc2.env import environment
from pysc2.env import sc2_env
from pysc2.lib import features
from pysc2.lib import actions

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3
_PLAYER_HOSTILE = 4

_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_NOT_QUEUED = [0]
_SELECT_ALL = [0]

class mineral(object):
    def __init__(self):
        self.unchoose =True

    def step(self,obs):
        screen = np.array(obs.observation['screen'],dtype=np.float32)
        if self.unchoose == True:
            return actions.FunctionCall(_SELECT_ARMY,[_SELECT_ALL])
        a =np.random.randint(0,63,[64,64])
        return actions.FunctionCall(_MOVE_SCREEN,['screen',a])

sc2_env.SC2Env(map_name="CollectMineralShards")