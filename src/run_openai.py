"""
This code uses the OpenAI baselines to learn the policies.
However, the current implementation ignores the LTL formula.
I left this code here as a reference and for debugging purposes.
"""

try:
    from mpi4py import MPI
except ImportError:
    MPI = None
import numpy as np
import tensorflow as tf
import gym, multiprocessing, sys, os, argparse
from baselines import deepq, bench, logger
from baselines.ppo2 import ppo2
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.models import get_network_builder
from baselines.common import set_global_seeds
import envs.gym_letters
import ltl_wrappers

def make_env(env_id, mpi_rank=0, subrank=0, seed=None, logger_dir=None, initializer=None):
    if initializer is not None:
        initializer(mpi_rank=mpi_rank, subrank=subrank)

    env = gym.make(env_id)

    # Adding general wraps
    env = ltl_wrappers.LTLEnv(env)
    env = ltl_wrappers.NoLTLWrapper(env) # For testing purposes

    env.seed(seed + subrank if seed is not None else None)
    env = bench.Monitor(env,
                        logger_dir and os.path.join(logger_dir, str(mpi_rank) + '.' + str(subrank)),
                        allow_early_resets=True)

    return env

def make_vec_env(env_id, num_env, seed, start_index=0, initializer=None, force_dummy=False):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari.
    """
    mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    seed = seed + 10000 * mpi_rank if seed is not None else None
    logger_dir = logger.get_dir()
    def make_thunk(rank, initializer=None):
        return lambda: make_env(
            env_id=env_id,
            mpi_rank=mpi_rank,
            subrank=rank,
            seed=seed,
            logger_dir=logger_dir,
            initializer=initializer
        )

    set_global_seeds(seed)
    if not force_dummy and num_env > 1:
        return SubprocVecEnv([make_thunk(i + start_index, initializer=initializer) for i in range(num_env)])
    else:
        return DummyVecEnv([make_thunk(i + start_index, initializer=None) for i in range(num_env)])

def build_env(env_id, agent, force_dummy=False, num_env=None, seed=None):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = num_env or ncpu

    if agent in ['dqn','trpo']:
        logger_dir = logger.get_dir()
        env = make_env(env_id, logger_dir=logger_dir)
    else:
        env = make_vec_env(env_id, nenv, seed, force_dummy=force_dummy)
        # NOTE:   this is a more efficient way to stack the last 4 frames, but it is not compatible with my memory modules :(
        # SOURCE: https://github.com/openai/baselines/issues/663
        #frame_stack_size = 4
        #env = VecFrameStack(env, frame_stack_size)

    return env

def learn_letters(agent, env):
    if agent == "dqn":
        model = deepq.learn(
            env,
            "mlp", num_layers=4, num_hidden=128, activation=tf.tanh, # tf.nn.relu
            hiddens=[128],
            dueling=True,
            lr=1e-5,
            total_timesteps=int(1e7),
            buffer_size=100000,
            batch_size=32,
            exploration_fraction=0.1,
            exploration_final_eps=0.1, #0.01, -> testing...
            train_freq=1,
            learning_starts=10000,
            target_network_update_freq=100,
            gamma=0.9,
            print_freq=50
        )

    elif "ppo" in agent:
        mlp_net = get_network_builder("mlp")(num_layers=5, num_hidden=128, activation=tf.tanh) # tf.nn.relu
        ppo_params = dict(
            nsteps=128,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            lr=1e-4,
            gamma=0.99, # Note that my results over the red/blue doors were computed using gamma=0.9!
            lam=0.95,
            log_interval=50,
            nminibatches=8,
            noptepochs=1,
            #save_interval=100,
            cliprange=0.2)
        if "lstm" in agent:
            # Adding a recurrent layer
            ppo_params["network"] = 'cnn_lstm'
            ppo_params["nlstm"] = 128
            ppo_params["conv_fn"] = mlp_net
            ppo_params["lr"] = 0.001
        else:
            # Using a standard MLP
            ppo_params["network"] = mlp_net

        timesteps=int(1e9)

        model = ppo2.learn(
            env=env,
            total_timesteps=timesteps,
            **ppo_params
        )
    else:
        assert False, agent + " hasn't been implemented yet"

    return model


def run_agent(agent, env_id, run_id):
    log_path  = "results/" + agent.upper() + "/" + env_id + "/" + str(run_id)
    save_path = log_path + "/trained-model"
    logger.configure(log_path)

    # Setting the number of workers
    num_env = 8

    # Creating the memory-based environments
    env = build_env(env_id, agent, num_env=num_env)

    # Running the agent
    model = learn_letters(agent, env)

    model.save(save_path)
    env.close()


if __name__ == '__main__':

    agent  = 'ppo'
    env_id = 'Letter-4x4-v0'
    run_id = 0
    run_agent(agent, env_id, run_id)
