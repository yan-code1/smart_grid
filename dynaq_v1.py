from algorithms.robot.dynaq_agent import DynaQAgent
from tqdm import tqdm
import config.global_exp_config as gc
from comm_env.env_move import Env
from helper.data_move_saver import DataSaver
CODE_VERSION = "DyanQ_move_v1"

def main():
    # 初始化对象
    saver = DataSaver(CODE_VERSION)  # 数据保存对象
    env = Env()  # 环境数据调用对象
    agent = DynaQAgent(10)  # QL对象
    # agent = QLearning(STATE_NUMBER,ACTION_NUMBER)
    for episode in range(gc.EPISODE_NUMBER):
        state = env.reset()  # 环境状态更新
        agent.reset()  # 智能体reset()


        for step in range(0, gc.STEP_NUMBER):
            action_next = agent.choose_action(state)
            state_next = env.step(action_next)
            agent.learn(state, state_next, action_next)
            saver.save_step(episode, step, state, state_next)
            # update
            state = state_next
            # save

        # save QT
        saver.save_qtable(episode, agent)
    # save .mat
    saver.save_mat()
if __name__ == '__main__':
    main()
