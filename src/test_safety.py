import argparse
import time
import sys
import numpy as np
import glfw
import utils
import torch

import gym
import safety_gym
import ltl_wrappers
import ltl_progression
from gym import wrappers, logger
from envs.safety import safety_wrappers

class RandomAgent(object):
    """This agent picks actions randomly"""
    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, obs):
        return self.action_space.sample()

class PlayAgent(object):
    """
    This agent allows user to play with Safety's Point agent.

    Use the UP and DOWN arrows to move forward and back and
    use '<' and '>' to rotate the agent.
    """
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.prev_act = np.array([0, 0])
        self.last_obs = None

    def get_action(self, obs):
        # obs = obs["features"]

        key = self.env.key_pressed

        if(key == glfw.KEY_COMMA):
            current = np.array([0, 0.4])
        elif(key == glfw.KEY_PERIOD):
            current = np.array([0, -0.4])
        elif(key == glfw.KEY_UP):
            current = np.array([0.1, 0])
        elif(key == glfw.KEY_DOWN):
            current = np.array([-0.1, 0])
        elif(key == -1): # This is glfw.RELEASE
            current = np.array([0, 0])
            self.prev_act = np.array([0, 0])
        else:
            current = np.array([0, 0])

        self.prev_act = np.clip(self.prev_act + current, -1, 1)

        return self.prev_act

def run_policy(agent, env, max_ep_len=None, num_episodes=100, render=True):
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(1) #########
    o, r, d, ep_ret, ep_cost, ep_len, n = env.reset(), 0, False, 0, 0, 0, 0
    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)

        ltl_goal = ltl_progression.spotify(env.ltl_goal)
        env.show_text(ltl_goal.to_str())
        if("progress_info" in o.keys()):
            env.show_prog_info(o["progress_info"])

        a = agent.get_action(o)
        a = np.clip(a, env.action_space.low, env.action_space.high)
        o, r, d, info = env.step(a)
        ep_ret += r
        ep_cost += info.get('cost', 0)
        ep_len += 1

        if d or (ep_len == max_ep_len):
            o, r, d, ep_ret, ep_cost, ep_len = env.reset(), 0, False, 0, 0, 0
            n += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    subparsers = parser.add_subparsers(dest='command')

    parser.add_argument('env_id', default='SafexpTest-v0', help='Select the environment to run')

    parser_play = subparsers.add_parser('play',   help='A playable agent that can be controlled.')
    parser_random = subparsers.add_parser('random', help='An agent that picks actions at random (for testing).')
    parser_viz = subparsers.add_parser('viz', help='Load the agent model from a file and visualize its action on the env.')

    parser_viz.add_argument('model_path', type=str, help='The path to the model to load.')

    parser_viz.add_argument("--ltl-sampler", default="Default",
                    help="the ltl formula template to sample from (default: DefaultSampler)")


    args = vars(parser.parse_args()) # make it a dictionary
    outdir = './storage/random-agent-results'

    if (args["command"] == "play"):
        env = gym.make(args["env_id"])
        env.num_steps = 10000000
        env = safety_wrappers.Play(env)
        env = ltl_wrappers.LTLEnv(env, ltl_sampler="Default")

        agent = PlayAgent(env)

    elif (args["command"] == "random"):
        env = gym.make(args["env_id"])
        env.num_steps = 10000
        env = safety_wrappers.Play(env)
        env = ltl_wrappers.LTLEnv(env, ltl_sampler="Default")

        agent = RandomAgent(env.action_space)

    elif (args["command"] == "viz"):
        # If the config is available (from trainig) then just load it here instead of asking the user of this script to provide all training time configs
        config = vars(utils.load_config(args["model_path"]))
        args.update(config)

        env = gym.make(args["env_id"])
        env = safety_wrappers.Play(env)
        env = ltl_wrappers.LTLEnv(env, ltl_sampler=args["ltl_sampler"], progression_mode=args["progression_mode"])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        agent = utils.Agent(env, env.observation_space, env.action_space, args["model_path"],
                args["ignoreLTL"], args["progression_mode"], args["gnn"], device=device, dumb_ac=args["dumb_ac"])
    else:
        print("Incorrect command: ", args["command"])
        exit(1)

    run_policy(agent, env, max_ep_len=30000, num_episodes=1000)

