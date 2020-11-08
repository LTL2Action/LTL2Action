"""
This class defines the environments that we are going to use.
Note that this is the place to include the right LTL-Wrapper for each environment.
"""


import gym
import gym_minigrid
import envs.gym_letters
import ltl_wrappers

def make_env(env_key, progression_mode, ltl_sampler, seed=None, intrinsic=0):
    """
    Only support for letter envs for now
    """
    env = gym.make(env_key)
    env.seed(seed)

    # Adding LTL wrappers
    env = ltl_wrappers.LTLLetterEnv(env, progression_mode, ltl_sampler, intrinsic)

    return env
