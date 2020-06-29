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

    env.seed(seed + subrank if seed is not None else None)
    env = bench.Monitor(env,
                        logger_dir and os.path.join(logger_dir, str(mpi_rank) + '.' + str(subrank)),
                        allow_early_resets=True)
    
    # Adding general wraps
    env = ltl_wrappers.LTLLetterEnv(env)

    return env

def make_vec_env(env_id, num_env, seed, start_index=0, reward_scale=1.0, initializer=None, force_dummy=False):
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

def learn_letters(agent, env, env_id):
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
            lr=1e-5,#1e-5, # 0.001 might work better!
            gamma=0.99, # Note that my results over the red/blue doors were computed using gamma=0.9!
            lam=0.95,
            log_interval=10,
            nminibatches=8,
            noptepochs=4,
            #save_interval=100,
            cliprange=0.2)
        if "lstm" in agent:
            # Adding a recurrent layer
            ppo_params["network"] = 'cnn_lstm'
            ppo_params["nlstm"] = 128
            ppo_params["conv_fn"] = mlp_net
            if "MemoryS" in env_id:
                # NOTE: this learning rate works better in the MemoryS envs for ppo-lstm
                ppo_params["lr"] = 0.001
        else:
            # Using a standard MLP
            ppo_params["network"] = mlp_net
            
        if "RedBlueDoors" in env_id:
            timesteps=int(1e8)
        elif "MemoryS" in env_id:
            timesteps=int(5e8)
        else:
            assert False

        model = ppo2.learn(
            env=env, 
            total_timesteps=timesteps, 
            **ppo_params
        )
    else:
        assert False, agent + " hasn't been implemented yet"

    return model


def run_agent(agent, env_id, run_id):
    exp_name = env_id + '-' + mem_type.upper() + str(mem_size)
    log_path  = "results/" + agent.upper() + "/" + exp_name + "/" + str(run_id)
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



def test_env():
    # Creating the memory-based environments
    env = build_env("Letter-4x4-v0", "dqn")

    import random
    for _ in range(10):
        obs = env.reset()
        print(env.action_space)
        print(env.observation_space)
        print(obs["ltl"])
        print(np.array(obs["features"]).shape)
        input()
        #print(obs.count(), len(obs), np.array(obs).shape)
        #input()
        for _ in range(10000):
            a = random.randrange(env.action_space.n)
            obs, reward, done, info = env.step(a%env.action_space.n)

            print(obs["ltl"])
            print(np.array(obs["features"]).shape)
            print(reward)

            if done:
                break

    env.close()
    exit()


if __name__ == '__main__':

    test_env()
    #test_random_agent()
    #env_id = 'PongDeterministic-v0'   # with sticky actions
    #env_id = 'PongDeterministic-v4'  # w/o sticky actions

    # EXAMPLE: python run.py --agent='ppo-lstm' --mem_type='n' --mem_size=1 --env_id='MiniGrid-RedBlueDoors-8x8-v0' --run_id=99
    # EXAMPLE: python run.py --agent='ppo-lstm' --mem_type='n' --mem_size=1 --env_id='MiniGrid-MemoryS7-v0' --run_id=99
    # EXAMPLE: python run.py --agent='ppo' --mem_type='s' --mem_size=6 --env_id='MiniGrid-MemoryS7-v0' --run_id=93

    # Running:
    # EXAMPLE: python run.py --agent='dqn' --mem_type='n' --mem_size=1 --env_id='MiniGrid-RedBlueDoors-6x6-v0' --run_id=1
    # EXAMPLE: python run.py --agent='dqn' --mem_type='b' --mem_size=3 --env_id='MiniGrid-RedBlueDoors-6x6-v0' --run_id=1
    
    # EXAMPLE: python run.py --agent='dqn' --mem_type='n' --mem_size=1 --env_id='MiniGrid-RedBlueDoors-8x8-v0' --run_id=1
    # EXAMPLE: python run.py --agent='dqn' --mem_type='k' --mem_size=4 --env_id='MiniGrid-RedBlueDoors-8x8-v0' --run_id=1
    # EXAMPLE: python run.py --agent='dqn' --mem_type='b' --mem_size=3 --env_id='MiniGrid-RedBlueDoors-8x8-v0' --run_id=1
    # EXAMPLE: python run.py --agent='dqn' --mem_type='s' --mem_size=3 --env_id='MiniGrid-RedBlueDoors-8x8-v0' --run_id=1

    # TODO:
    # EXAMPLE: python run.py --agent='ppo' --mem_type='n' --mem_size=1 --env_id='MiniGrid-RedBlueDoors-6x6-v0' --run_id=1
    # EXAMPLE: python run.py --agent='ppo' --mem_type='k' --mem_size=4 --env_id='MiniGrid-RedBlueDoors-6x6-v0' --run_id=1
    # EXAMPLE: python run.py --agent='ppo' --mem_type='b' --mem_size=3 --env_id='MiniGrid-RedBlueDoors-6x6-v0' --run_id=1
    # EXAMPLE: python run.py --agent='ppo' --mem_type='s' --mem_size=3 --env_id='MiniGrid-RedBlueDoors-6x6-v0' --run_id=1

    # python run.py --agent='ppo-lstm' --mem_type='n' --mem_size=1 --env_id='Hallway-Cookies-v0' --run_id=22

    

    # Getting params
    """
    agents   = ['dqn','ppo', 'ppo-lstm', 'a3c', 'a3c-lstm', 'acer', 'acer-lstm', 'trpo']
    memories = ['n', 'k', 'b', 's','sas', 'm']

    parser = argparse.ArgumentParser(prog="run", description='Runs the selected RL agent over the selected world.')
    parser.add_argument('--agent', default='qlearning', type=str, 
                        help='This parameter indicates which RL algorithm to use. The options are: ' + str(agents))
    parser.add_argument('--mem_type', default='s', type=str, 
                        help='This parameter indicates which type of memory to use. The options are: ' + str(memories))
    parser.add_argument('--mem_size', default=1, type=int, 
                        help='This parameter indicates the size of the memory.')
    parser.add_argument('--env_id', default='PongDeterministic-v0', type=str)
    parser.add_argument('--run_id', default=1, type=int, 
                        help='This parameter indicates the run id.')


    args = parser.parse_args()
    assert args.agent in agents, "Agent " + args.algorithm + " hasn't been implemented yet"
    assert args.mem_type in memories, "Memory " + args.mem_type + " hasn't been defined yet"
    assert args.mem_size >= 0, "The size of the memory must be non-negative"
    assert args.run_id >= 0, "The run id must be non-negative"


    # Running the experiment
    agent    = args.agent    #['dqn','ppo', 'ppo-lstm', 'a3c', 'a3c-lstm', 'acer', 'acer-lstm']
    mem_type = args.mem_type #['nmem', 'kmem', 'bmem', 'smem']
    mem_size = args.mem_size
    env_id   = args.env_id
    run_id   = args.run_id

    print("Running", agent, mem_type, mem_size, env_id, run_id)
    input()

    run_agent(agent, mem_type, mem_size, env_id, run_id)
    """