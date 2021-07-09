from src.agents import Agent, ReplayDDQNAgent, PriorityReplayDDQNAgent
from src.environment_utils import Execution_Manager


STATES = 37
ACTIONS = 4
LAYERS = [64, 64]
ENV_PATH = "/Users/claudiocoppola/code/deep-reinforcement-learning/p1_navigation/Banana.app"

package_path = '/'.join(__file__.split('/')[:-1])

out_file = 'priority_replay.pth'

if __name__ == '__main__':

    agent = PriorityReplayDDQNAgent(STATES, ACTIONS, hidden_layers=LAYERS, seed=0)

    training_manager = Execution_Manager(agent, ENV_PATH)

    training_manager.dqn_train(
        n_episodes=1800, max_t=300,
        eps_start=1.0, eps_end=0.01, eps_decay=0.990,
        save_on_score=13., out_file=package_path + '/assets/models/{}'.format(out_file))

    training_manager.plot_scores()

    training_manager.play_episode(200)