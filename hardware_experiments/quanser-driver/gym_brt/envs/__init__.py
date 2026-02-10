from gym_brt.envs.reinforcementlearning_extensions.rl_reward_functions import (
    swing_up_reward,
    balance_reward
)

from gym_brt.envs.qube_balance_env import (
    QubeBalanceEnv,
)

from gym_brt.envs.qube_swingup_env import (
    QubeSwingupEnv,
)

from gym_brt.envs.reinforcementlearning_extensions.rl_gym_classes import (
    QubeBeginUpEnv,
    QubeBeginDownEnv,
    RandomStartEnv,
    NoisyEnv,
    convert_state,
    convert_state_back
)

from gym.envs.registration import register

register(
    id='QubeBeginDownEnv-v1',
    entry_point='gym_brt.envs:QubeBeginDownEnv',
    )

register(
    id='QubeSwingupEnv-v1',
    entry_point='gym_brt.envs:QubeSwingupEnv',
    )

register(
    id='QubeBeginUpEnv-v1',
    entry_point='gym_brt.envs:QubeBeginUpEnv',
    )

register(
    id='QubeNoisyEnv-v1',
    entry_point='gym_brt.envs:NoisyEnv',
    )

register(
    id='QubeRandomStartEnv-v1',
    entry_point='gym_brt.envs:RandomStartEnv',
    )