==============
supermario-dqn
==============

Deep Reinforcement Learning Agent for Super Mario Bros using OpenAI gym
[gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros) environment

## Usage

~~~shell
# create virtual env in the project folder
$ python3 -m venv .venv
$ source .venv/bin/activate

# install dependencies
$ pip3 install -r requirements

# use
$ supermario_train --help
usage: supermario_train [-h] [--batch_size BATCH_SIZE]
                        [--fit_interval FIT_INTERVAL] [--gamma GAMMA]
                        [--eps_start EPS_START] [--eps_end EPS_END]
                        [--target_update TARGET_UPDATE]
                        [--save_interval SAVE_INTERVAL]
                        [--save_path SAVE_PATH] [--memory_size MEMORY_SIZE]
                        [--num_episodes NUM_EPISODES] [--verbose VERBOSE]
                        [--load LOAD] [--log_file_dir LOG_FILE_DIR]
                        [--finally_show]

Handle training

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        size of each batch used for training
  --fit_interval FIT_INTERVAL
                        fit every `fit_interval` examples available
  --gamma GAMMA         discount rate used for Q-values learning
  --eps_start EPS_START
                        start probability to choose a random action
  --eps_end EPS_END     end probability to choose a random action
  --target_update TARGET_UPDATE
                        number of episodes between each target dqn update
  --save_interval SAVE_INTERVAL
                        number of episodes between each network checkpoint
  --save_path SAVE_PATH
                        where save trained model
  --memory_size MEMORY_SIZE
                        size of replay memory
  --num_episodes NUM_EPISODES
                        number of games to be played before end
  --verbose VERBOSE     verbosity of output
  --load LOAD           load a saved state_dict
  --log_file_dir LOG_FILE_DIR
                        file path where write logs
  --finally_show        finally show a play

# and finally

$ supermario_play --help
usage: play a game [-h] [--world_stage WORLD_STAGE WORLD_STAGE] model

positional arguments:
  model                 neural network model

optional arguments:
  -h, --help            show this help message and exit
  --world_stage WORLD_STAGE WORLD_STAGE
                        select a specific world and stage, world in [1..8],
                        stage in [1..4]

~~~

For now I have only performed the training for the first stage of level 1 with the following results
(about 3 hours and 2000 episodes)

![][trained/train1/rewards_over_steps.png]
