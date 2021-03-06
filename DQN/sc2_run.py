from dueling_dqn import Mineral
from utils import move
import numpy as np
import sys
from absl import flags

from pysc2.env import sc2_env
from pysc2.env import environment
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

#DQN
N_ACTIONS = 4
N_FATURES = 64*64
LEARNING_RATE = 0.001
REWARN_DECAY = 1
E_GREED = 0.9
MEMARY_SIZE = 20000
REPLACE_TARGET_ITER = 100
BATCH_SIZE = 32

EPSIODES = 10000
#有错误无法收敛
FLAGS = flags.FLAGS
FLAGS(sys.argv)
with sc2_env.SC2Env(map_name="CollectMineralShards",visualize=True,step_mul=1) as env:
    RL = Mineral(n_actions=4,n_features=64*64,
                 learning_rate=LEARNING_RATE,
                 reward_decay=REWARN_DECAY,
                 e_greed=E_GREED,
                 memory_size=MEMARY_SIZE,
                 batch_size=BATCH_SIZE,
                 replace_target_iter=REPLACE_TARGET_ITER,
                 output_graph=True)
    step_global =0

    for i in range(EPSIODES):
        env.reset()
        step=0
        obs =env.step(actions=[actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])])
        state =obs[0].observation["screen"][_PLAYER_RELATIVE]
        player_y, player_x = (state == _PLAYER_FRIENDLY).nonzero()
        player_center = [int(player_x.mean()), int(player_y.mean())]
        target = player_center
        total_reward =0
        reward = 0
        state = np.reshape(state,[64*64])
        while True:
            if step%5==0:
                action = RL.choose_action(state)
                target = move(target,action,3)
                obs = env.step(actions=[actions.FunctionCall(_MOVE_SCREEN,[_NOT_QUEUED,target])])
                state_ = np.array(obs[0].observation["screen"][_PLAYER_RELATIVE])
                state_ = np.reshape(state_,[64*64])
                reward += obs[0].reward
                reward1 = obs[0].reward
                RL.store_transition(state,action,reward,state_)
                if (step_global>1000) and (step_global%4==0):
                    RL.learn()
                state = state_
                reward=0
            else:
                obs=env.step(actions=[actions.FunctionCall(_NO_OP,[])])
                reward += obs[0].reward
                reward1 = obs[0].reward

            total_reward+=reward1

            done = obs[0].step_type == environment.StepType.LAST
            if done:
                print(total_reward)
                break

            step+=1
            step_global+=1
