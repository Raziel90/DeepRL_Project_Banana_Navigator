
from collections import deque
import matplotlib.pyplot as plt
import numpy as np

import torch
from unityagents import UnityEnvironment


class Execution_Manager():
    def __init__(self, agent, unity_env_path="/data/Banana_Linux_NoVis/Banana.x86_64"):

        self.env = UnityEnvironment(file_name=unity_env_path)
        # get the default brain
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        self.agent = agent
        self.train_scores = []

    def plot_scores(self, scores=None):
        scores = self.train_scores if scores is None else scores
        if len(scores) > 0:
            # plot the scores
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.plot(np.arange(len(scores)), scores)
            plt.ylabel('Score')
            plt.xlabel('Episode #')
            plt.show()

    def play_episode(self, max_t, eps, train_mode=False):
        env_info = self.env.reset(train_mode=train_mode)[self.brain_name] # reset the environment
        state = env_info.vector_observations[0]     
        score = 0
        for t in range(max_t):
            action = self.agent.act(state, eps)
            env_info = self.env.step(action)[self.brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]   
            self.agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 

        return score

    def exit(self):
        self.env.close()

    def dqn_train(self, n_episodes=5000, max_t=100, eps_start=1.0, eps_end=0.01, eps_decay=0.990, save_on_score=13., out_file='checkpoint.pth'):
        """Deep Q-Learning.
        
        Params
        ======
            n_episodes (int): maximum number of training episodes
            max_t (int): maximum number of timesteps per episode
            eps_start (float): starting value of epsilon, for epsilon-greedy action selection
            eps_end (float): minimum value of epsilon
            eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        """
        
        scores_window = deque(maxlen=100)  # last 100 scores
        eps = eps_start                    # initialize epsilon
        for i_episode in range(1, n_episodes+1):
            score =self.play_episode(max_t, eps, train_mode=True)
            scores_window.append(score)       # save most recent score
            self.train_scores.append(score)              # save most recent score
            eps = max(eps_end, eps_decay * eps) # decrease epsilon
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}\tEpisode Length: {:.2f}'.format(i_episode, np.mean(scores_window), max_t))
                # max_t = int(1.1 * max_t)
            if np.mean(scores_window) >= save_on_score:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
                self.agent.save_model(out_file)
                # torch.save(self.agent.qnetwork_local.state_dict(), out_file)
                break
        if np.mean(scores_window) < save_on_score:
            print('\nEnvironment not solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            self.agent.save_model(out_file)
            # torch.save(self.agent.qnetwork_local.state_dict(), out_file)      
        return self.train_scores




