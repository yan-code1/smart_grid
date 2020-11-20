from algorithms.robot.Dueling_DQN_agent import Due_DQNAgent
from tqdm import tqdm
import config.global_exp_config as gc
from comm_env.env_move import Env
from helper.data_move_saver import DataSaver
import config.env_robot_config as ec
CODE_VERSION = "Duel_DQN_move_v1"


def main():
    saver = DataSaver(CODE_VERSION)
    kwargs = {
        'gamma': 0,
        'epsi_high': 0.85,
        'epsi_low': 0.,
        'decay': 100,
        'lr': 0.005,
        'capacity': 1000,
        'batch_size': 32,
        'state_space_dim': 5,
        'action_space_dim': ec.ACTION_NUM
    }
    env = Env()
    Duel_DQN_Agent = Due_DQNAgent(**kwargs)
    for episode in range(gc.EPISODE_NUMBER):
        state_results = env.reset()
        Duel_DQN_Agent.reset()
        for step in range(0, gc.STEP_NUMBER):

            action_idx = Duel_DQN_Agent.choose_action(state_results)
            state_results_next = env.step(action_idx)
            Duel_DQN_Agent.learn(state_results, state_results_next, action_idx)
            saver.save_step(episode, step,state_results, state_results_next )

            state_results = state_results_next

        saver.save_mat()

if __name__ == '__main__':
    main()
