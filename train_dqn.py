from unityagents import UnityEnvironment
from src.agents import Agent, ReplayDDQNAgent, PriorityReplayDDQNAgent
from src.environment_utils import Execution_Manager
from random import randint
from src import package_path

STATES = 37
ACTIONS = 4
LAYERS = [64, 128, 64]
ENV_PATH = package_path + "/unity/Banana.app"

out_file = 'trained_model.pth'


checkpoint_path = package_path + '/assets/models/{}'.format(out_file)
plot_fig_path = package_path + '/assets/figs/{}'.format(out_file.split('.')[0] + '.svg')

if __name__ == '__main__':

    unity_env = UnityEnvironment(ENV_PATH, seed=randint(0, 1e6))
    brain_name = unity_env.brain_names[0]
    brain = unity_env.brains[brain_name]
    env_info = unity_env.reset(train_mode=True)[brain_name]
    states_dim = len(env_info.vector_observations[0])
    action_dim = brain.vector_action_space_size
    
    agent = PriorityReplayDDQNAgent(
        states_dim, action_dim, hidden_layers=LAYERS,
        update_every=4,
        lr=5e-4, # learning rate
        batch_size=64,
        buffer_size=int(1e5),
        dueling=True, # using Dueling DQN
        gamma=0.995, #discount factor
        tau=1e-3, # for soft update of target parameters (DDQN)
        priority_probability_a=.9, # Coefficient used to compute the importance of the priority weights during buffer sampling
        priority_correction_b=1. # Corrective factor for the loss in case of Priority Replay Buffer
        )

    training_manager = Execution_Manager(agent, unity_env)

    training_manager.dqn_train(
        n_episodes=1800, max_t=600,
        eps_start=1., eps_end=0.02, eps_decay=0.9950,
        target_score=13., out_file=checkpoint_path)

    training_manager.plot_scores(out_file=plot_fig_path)

    score = training_manager.play_episode(600)
    print('Played test episode with score: {:.2f}'.format(score))