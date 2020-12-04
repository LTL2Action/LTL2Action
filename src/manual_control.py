#!/usr/bin/env python3

import time
import argparse
import numpy as np
import gym
import gym_minigrid
import ltl_wrappers
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window

from envs.minigrid.adversarial import *


def redraw(img):
    if not args.agent_view:
        img = base_env.render(mode='rgb_array', tile_size=args.tile_size)

    window.show_img(img)

def reset():
    if args.seed != -1:
        ltl_env.seed(args.seed)

    obs = ltl_env.reset()

    window.set_caption(ltl_env.ltl_goal)

    redraw(obs)

def step(action):
    obs, reward, done, info = ltl_env.step(action)
    window.set_caption(ltl_env.ltl_goal)
    print('step=%s, reward=%.2f' % (base_env.step_count, reward))

    if done:
        print('done!')
        reset()
    else:
        redraw(obs)

def key_handler(event):
    print('pressed', event.key)

    if event.key == 'escape':
        window.close()
        return

    if event.key == 'backspace':
        reset()
        return

    if event.key == 'left':
        step(base_env.actions.left)
        return
    if event.key == 'right':
        step(base_env.actions.right)
        return
    if event.key == 'up':
        step(base_env.actions.forward)
        return

    # Spacebar
    if event.key == ' ':
        step(base_env.actions.toggle)
        return
    if event.key == 'pageup':
        step(base_env.actions.pickup)
        return
    if event.key == 'pagedown':
        step(base_env.actions.drop)
        return

    if event.key == 'enter':
        step(base_env.actions.done)
        return

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    help="gym environment to load",
    default='MiniGrid-MultiRoom-N6-v0'
)
parser.add_argument(
    "--seed",
    type=int,
    help="random seed to generate the environment with",
    default=-1
)
parser.add_argument(
    "--tile_size",
    type=int,
    help="size at which to render tiles",
    default=32
)
parser.add_argument(
    '--agent_view',
    default=False,
    help="draw the agent sees (partially observable view)",
    action='store_true'
)

args = parser.parse_args()

# `base_env` is the backend minigrid
# `env` is the (1-level) wrapped minigrid from our code
# `ltl_env` is the (2-level) wrapped minigrid with LTL goals
env = gym.make(args.env)
base_env = env.env
ltl_env = ltl_wrappers.LTLEnv(env, progression_mode="full", ltl_sampler="AdversarialSampler")

window = Window('gym_minigrid - ' + args.env)
window.reg_key_handler(key_handler)

reset()

# Blocking event loop
window.show(block=True)
