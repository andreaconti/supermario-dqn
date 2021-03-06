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
$ supermario_train -h
usage: supermario_train [-h] [--batch_size BATCH_SIZE]
                        [--fit_interval FIT_INTERVAL] [--gamma GAMMA]
                        [--eps_start EPS_START] [--eps_end EPS_END]
                        [--eps_decay EPS_DECAY]
                        [--target_update TARGET_UPDATE]
                        [--save_path SAVE_PATH] [--memory_size MEMORY_SIZE]
                        [--num_episodes NUM_EPISODES] [--resume RESUME]
                        [--checkpoint CHECKPOINT] [--random] [--render]
                        [--world_stage WORLD_STAGE WORLD_STAGE]
                        [--actions ACTIONS] [--test TEST] [--log]

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
  --eps_decay EPS_DECAY
                        decay of eps probabilities
  --target_update TARGET_UPDATE
                        number of episodes between each target dqn update
  --save_path SAVE_PATH
                        where save trained model
  --memory_size MEMORY_SIZE
                        size of replay memory
  --num_episodes NUM_EPISODES
                        number of games to be played before end
  --resume RESUME       load from a checkpoint
  --checkpoint CHECKPOINT
                        number of episodes between each network checkpoint
  --random              choose randomly different worlds and stages
  --render              rendering of frames, only for debug
  --world_stage WORLD_STAGE WORLD_STAGE
                        select specific world and stage
  --actions ACTIONS     select actions used between ["simple"]
  --test TEST           each `test` episodes network is used and tested over
                        an episode
  --log                 logs episodes results

# play
$ supermario_play -h
usage: play a game [-h] [--world_stage WORLD_STAGE WORLD_STAGE] [--skip SKIP]
                   [--processed]
                   model

positional arguments:
  model                 neural network model

optional arguments:
  -h, --help            show this help message and exit
  --world_stage WORLD_STAGE WORLD_STAGE
                        select a specific world and stage, world in [1..8],
                        stage in [1..4]
  --skip SKIP           number of frames to skip
  --processed           shows frames processed for neural network
~~~

### Results

~~~bash
$ supermario_play --skip 5 --world_stage 1 1 trained/train_1_1/model.pt
~~~

| rewards | play gif |
|---------|----------|
|![](trained/train_1_1/rewards_over_steps.png)| ![](trained/train_1_1/play_gif.png)|
