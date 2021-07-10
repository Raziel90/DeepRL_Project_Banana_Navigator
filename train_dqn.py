from unityagents import UnityEnvironment
from src.agents import Agent, ReplayDDQNAgent, PriorityReplayDDQNAgent
from src.environment_utils import Execution_Manager

from src import package_path

STATES = 37
ACTIONS = 4
LAYERS = [64, 64]
ENV_PATH = "/Users/claudiocoppola/code/deep-reinforcement-learning/p1_navigation/Banana.app"



out_file = 'priority_replay.pth'
checkpoint_path = package_path + '/assets/models/{}'.format(out_file)
plot_fig_path = package_path + '/assets/figs/{}'.format(out_file.split('.')[0] + '.svg')

if __name__ == '__main__':

    unity_env = UnityEnvironment(ENV_PATH)
    brain_name = unity_env.brain_names[0]
    brain = unity_env.brains[brain_name]
    env_info = unity_env.reset(train_mode=True)[brain_name]
    states_dim = len(env_info.vector_observations[0])
    action_dim = brain.vector_action_space_size

    agent = PriorityReplayDDQNAgent(states_dim, action_dim, hidden_layers=LAYERS)

    training_manager = Execution_Manager(agent, unity_env)

    training_manager.dqn_train(
        n_episodes=1800, max_t=400,
        eps_start=1.0, eps_end=0.01, eps_decay=0.990,
        save_on_score=20., out_file=checkpoint_path)

    training_manager.plot_scores(out_file=plot_fig_path)

    training_manager.play_episode(200)