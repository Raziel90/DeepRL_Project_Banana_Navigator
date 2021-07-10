from unityagents import UnityEnvironment
from src.agents import Agent, ReplayDDQNAgent, PriorityReplayDDQNAgent
from src.environment_utils import Execution_Manager

from src import package_path


LAYERS = [64, 64]
ENV_PATH = "/Users/claudiocoppola/code/deep-reinforcement-learning/p1_navigation/Banana.app"

# package_path = '/'.join(__file__.split('/')[:-1])

in_file = 'priority_replay.pth'
checkpoint_path = package_path + '/assets/models/{}'.format(in_file)
# plot_fig_path = package_path + '/assets/figs/{}'.format(out_file.split('.')[0] + '.svg')

if __name__ == '__main__':

    print(package_path)

    agent = PriorityReplayDDQNAgent.from_file(checkpoint_path)
    # agent = PriorityReplayDDQNAgent(states_dim, action_dim, hidden_layers=LAYERS, seed=0)

    training_manager = Execution_Manager(agent, ENV_PATH)



    score = training_manager.play_episode(200, eps=0.0)
    print('Episode Terminated with score: {}'.format(score))
    
    training_manager.exit()
