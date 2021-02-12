# ltl2action


## Installation instructions


We recommend using Python 3.6 to run this code.

1. `pip install -r requirements.txt`
2. Install Spot-2.9 (https://spot.lrde.epita.fr/install.html)
    - Follow the installation instructions at the link. Spot should be installed in `/usr/local/lib/python3.6/site-packages/spot`. This step usually takes around 20 mins.
3. (Optional) To run the OpenAI Safety Gym, you will need Mujoco installed, as well as an active license (http://www.mujoco.org/index.html). 
    - `pip install mujoco-py==2.0.2.9`
    - `pip install -e src/envs/safety/safety-gym/`

## Training RL agents

Training is done via `train_agent.py`, where you can specify the environment, the set of tasks to be performed, the encoding method for the LTL instructions, PPO hyperparameters, etc. See the beginning of `train_agent.py` for more details on all the arguments.

**To use pretrained LTL modules**, add the option `--pretrained-gnn`. It will automatically search `src/symbol-storage` for a compatible LTL module. You must first train a good policy (on any environment -- use LTLBootcamp for fast training) and save the model (done automatically). The default storage location for saved models is `src/storage` and you need to manually `cp -r` the model directory to `src/symbol-storage`. 

The specific commands used for our experiments are listed below (prepend commands with `python3`). We will assume you are running the command from within the `src/` directory.  

### LetterWorld
GNN+progression: `train_agent.py --algo ppo --env Letter-7x7-v3 --log-interval 5 --save-interval 20 --frames 20000000 --discount 0.94 --ltl-sampler Eventually_1_5_1_4 --epochs 4 --lr 0.0003`

GRU+progression: `train_agent.py --algo ppo --env Letter-7x7-v3 --log-interval 5 --save-interval 20 --frames 20000000 --discount 0.94 --ltl-sampler Eventually_1_5_1_4 --epochs 8 --lr 0.0003 --gnn GRU`

LSTM+progression: `train_agent.py --algo ppo --env Letter-7x7-v3 --log-interval 5 --save-interval 20 --frames 20000000 --discount 0.94 --ltl-sampler Eventually_1_5_1_4 --epochs 8 --lr 0.0003 --gnn LSTM`

GRU (without progression): `train_agent.py --algo ppo --env Letter-7x7-v3 --log-interval 5 --save-interval 20 --frames 20000000 --discount 0.94 --ltl-sampler Eventually_1_5_1_4 --recurrence 4 --progression-mode none --gnn GRU --batch-size 1024`

No-LTL: `train_agent.py --algo ppo --env Letter-7x7-v3 --log-interval 5 --save-interval 20 --frames 20000000 --discount 0.94 --ltl-sampler Until_1_3_1_2 --recurrence 4 --ignoreLTL --batch-size 1024`

Myopic: `train_agent.py --algo ppo --env Letter-7x7-v3 --log-interval 5 --save-interval 100 --frames 20000000 --discount 0.94 --ltl-sampler Eventually_1_5_1_4 --epochs 8 --lr 0.001 --progression-mode partial`

### LetterWorld (with pretrained LTL modules)
Use the same hyperparameters as the non-pretrained counterpart (reported above) and add the `--pretrained-gnn` option (also works for GRU, LSTM, etc). 

### ZoneEnv (requires Safety Gym/Mujoco)
Myopic: `train_agent.py --algo ppo --env Zones-5-v0 --gnn GRU --progression-mode partial --ltl-sampler Until_1_2_1_1 --frames 22000000 --frames-per-proc 4096 --batch-size 1024 --lr 0.0003 --discount 0.998 --entropy-coef 0.003 --log-interval 1 --save-interval 2 --epochs 10`

GNN: `train_agent.py --also ppo --env Zones-5-v0 --ltl-sampler Until_1_2_1_1 --frames-per-proc 4096 --batch-size 2048 --lr 0.0003 --discount 0.998 --entropy-coef 0.003 --log-interval 1 --save-interval 2 --epochs 10 --frames 22000000` (add `--pretrained-gnn` for pretrained version)

### MiniGrid (toy experiment)
GNN+progression: `train_agent.py --algo ppo --env Adversarial-v0 --ltl-sampler Adversarial --frames-per-proc 1024  --discount 0.96 --log-interval 1 --save-interval 50 --frames 3000000`

### LTLBootcamp (pretraining)
GNN+progression (Avoidance Tasks): `train_agent.py --algo ppo --env Simple-LTL-Env-v0 --log-interval 1 --save-interval 30 --frames-per-proc 512 --batch-size 1024 --frames 10000000 --dumb-ac --discount 0.9 --ltl-sampler Eventually_1_5_1_4 --lr 0.001 --clip-eps 0.1 --gae-lambda 0.5 --epochs 2`

GNN+progression (Partially-Ordered Tasks): `train_agent.py --algo ppo --env Simple-LTL-Env-v0 --log-interval 1 --save-interval 30 --frames-per-proc 512 --batch-size 1024 --frames 10000000 --dumb-ac --discount 0.9 --ltl-sampler Eventually_1_5_1_4 --lr 0.001 --clip-eps 0.1 --gae-lambda 0.5 --epochs 2`

GRU+progression (Avoidance Tasks): `train_agent.py --algo ppo --env Simple-LTL-Env-v0 --log-interval 1 --save-interval 30 --frames-per-proc 512 --batch-size 1024 --frames 10000000 --dumb-ac --discount 0.9 --ltl-sampler Until_1_3_1_2 --lr 0.001 --clip-eps 0.1 --gae-lambda 0.5 --gnn GRU --epochs 2`

GRU+progression (Partially-Ordered Tasks): `train_agent.py --algo ppo --env Simple-LTL-Env-v0 --log-interval 1 --save-interval 30 --frames-per-proc 512 --batch-size 1024 --frames 10000000 --dumb-ac --discount 0.9 --ltl-sampler Eventually_1_5_1_4 --lr 0.003 --gnn GRU --epochs 4`

    
## Evaluating trained agents

If you have a trained model for the ZoneEnv, you can use this command to run and visualize it. It automatically loads the same tasks as it was trained on. 

`python3 test_safety.py viz zone-random-agent/ Zones-5-v0`

You can evaluate trained policies using `utils/evaluator.py`. You need to specify the model path, which also supports the `*` wildcard. For example, the following command evaluates all models whose names start with `GRU` located in `storage/`. 
`python3 utils/evaluator.py --ltl-sampler Until_1_3_1_2 --model-path storage/GRU* --procs 16  --discount 0.94 --eval-episodes 100 --gnn GRU`. 

A Jupyter notebook with code for plotting experimental data is available in `plots/plotter.ipynb`.