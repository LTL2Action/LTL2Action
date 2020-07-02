import gym
import gym_minigrid
import envs.gym_letters
import ltl_wrappers

def make_env(env_key, seed=None):
    """
    Only support for letter envs for now
    """
    env = gym.make(env_key)
    env.seed(seed)
    
    # Adding LTL wrappers
    env = ltl_wrappers.LTLLetterEnv(env)

    return env

"""
def make_env(env_key, seed=None):
    env = gym.make(env_key)
    env.seed(seed)
    return env
"""
