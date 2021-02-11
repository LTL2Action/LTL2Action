# Instructions
# conda create -n ltl2action-environment python=3.6
# pip install gym
# pip install gym-minigrid
# pip install SymPy
# pip install tensorflow==1.14
# in baselines folder -> pip install -e .
# git clone https://github.com/lcswillems/rl-starter-files.git
# cd rl-starter-files
# pip install -r requirements.txt
# Install Spot-2.9 (https://spot.lrde.epita.fr/install.html)
#	- wget http://www.lrde.epita.fr/dload/spot/spot-2.9.tar.gz
#	- cd spot-2.9
#	- ./configure --prefix /u/home_folder/
#	- make
#	- make install


# Loading libraries
export PATH="/u/home_folder/anaconda3/bin:$PATH"
export PYTHONPATH=$PYTHONPATH:/u/home_folder/lib/python3.6/site-packages

# Activating environment
source activate ltl2action-environment


###############################
# conda create -n ltl2action-environment python=3.6
# pip install
