# ltl2action

This is the code repository accompanying the paper [**LTL2Action: Generalizing LTL Instructions for Multi-Task RL**](https://arxiv.org/abs/2102.06858). 

Our agent's behaviour, compared to a myopic one is demonstrated on ZoneEnv, a custom environment based on OpenAI's SafetyGym:
<p align="center">
    <img width="500" src="https://github.com/LTL2Action/LTL2Action/blob/master/zone_env.gif">
</p>

## Installation instructions


We recommend using Python 3.6 to run this code.

1. `pip install -r requirements.txt`
2. Install Spot-2.9 (https://spot.lrde.epita.fr/install.html)
    - Follow the installation instructions at the link. Spot should be installed in `/usr/local/lib/python3.6/site-packages/spot`. This step usually takes around 20 mins.
3. (Optional) To run the OpenAI Safety Gym, you will need Mujoco installed, as well as an active license (http://www.mujoco.org/index.html). 
    - `pip install mujoco-py==2.0.2.9`
    - `pip install -e src/envs/safety/safety-gym/`

## Training Agents

Instructions for training and evaluating RL agents on each of our domains is available in the [`src`](src/) folder.
