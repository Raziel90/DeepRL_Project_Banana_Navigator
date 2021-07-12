# BananaBrain Project Submission
Code of the Project submission for the DeepRL Udacity Course: Navigation.

The simulation contains a single agent that navigates a large environment.  At each time step, it has four actions at its disposal:
- `0` - walk forward 
- `1` - walk backward
- `2` - turn left
- `3` - turn right

The state space has `37` dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  A reward of `+1` is provided for collecting a yellow banana, and a reward of `-1` is provided for collecting a blue banana. 


The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

## Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the DRLND GitHub repository, in the `unity/` folder, and unzip (or decompress) the file. 
   
# Introduction

In this repo you can find the code to train and run AI trained though Deep Reinforcement Learning based on the DQN Architecture.
In particular, You'll find implementation of the following variants:
- Basic Agent (for execution only) `Agent`
- Double DQN with Classic Memory Replay `ReplayDDQNAgent`
- Double DQN with Priority Based Memory Replay `PriorityReplayDDQNAgent`

Each those support both the following architectures:
- DQN `DQN`
- Dueling DQN `Dueling_DQN`
  
## Repo Organisation
In the root of the folder you can find the following scripts:
```python
train_dqn.py # shapes the training procedure and, if appropriately edited, allows to select the hyperparameters of training and the type of architecture used
run_dqn.py # loads a trained agent and runes an episode of fixed length
```
In the `assets` folder, all the files resulting from the training process can be found:
- `figs` contains the plot figures obtained from the training process
- `models` contains the checkpoints of the trained agents.

In the `unity` folder, the compiled version of the unity enviroment should be placed.

In the `src` folder are located the core components of the models' and agents' code:
```python
model.qnet.py #contains the code of the DQN and Dueling DQN models.
replay_buffers.buffer_replay #contains the code regarding the 2 variants of buffer replay
agents.py #contains the code of the 3 implemented agent variants
environment_utils.py #contains the code that connects to the unity environment and that manages the training and execution of the agent.
```

# Agent Training
In this section, one can find out how to edit the training parameters and which have been chosen for this specific problem.
## Training Script
Most training details can be edited in the `train_dqn.py` file.
`LAYERS = [64, 128, 64]` indicates the size (and number) of the hidden layers of the chosen DQN (In this case 3 hidden layers of respectfully 64, 128 and 64 neurons).
`out_file = 'dueling_priority_replay.pth'` indicates the name of the file used for the checkpoint and the plot figure file.

Most of the training hyperparameters are decided here:
```python
 training_manager.dqn_train(
        n_episodes=1800, max_t=400,
        eps_start=1, eps_end=0.02, eps_decay=0.9950,
        target_score=13., out_file=checkpoint_path)
``` 
In which sets the max number of episodes `n_episodes`, maximum length of the episode `max_t`, value of epsilon at the start of the training `eps_start`, the minimum value of epsilon `eps_end`, the epsilon decay factor per episode `eps_decay`, the training target score (the checkpoint will be saved at scores higher than this) `target_score` and the path of the checkpoint file to save `outfile`.

An Additional set of hyper-parameters define the behaviour and the learning of the agent. Those can be included in the `kwargs` of the Agent class and are the following:
### For all the Agents Categories
```python
dueling # (boolean) if True uses a Dueling DQN architecture if False a simple DQN
```
### Exclusive to (ReplayDDQNAgent, PriorityReplayDDQNAgent)
```python
rl # learning rate of the neural network
batch_size # size of the batch used for training the network
gamma # discount factor used for the Agent Learning
update_every # defines after how many steps the network should be updated
buffer_size # size of the memory for the Memory Replay
tau # coefficient of (linear) interpolation between the local and target networks (used in the DDQN soft_update)
```
### Exclusive to (PriorityReplayDDQNAgent)
```python
priority_probability_a # Coefficient used to compute the importance of the priority weights during buffer sampling
priority_correction_b # Corrective factor for the loss in case of Priority Replay Buffer
```
