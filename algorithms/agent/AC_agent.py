
from algorithms.QL.actor_critic import ActorCritic
import config.env_robot_config as ec
import  config.global_exp_config as gc
class AC_agent:
    def __init__(self):
        self.ac = ActorCritic(weight_init=0.01, num_actions=ec.ACTION_NUM, gamma=0.9, alpha=0.01, beta1=0.005, epsilon=0.1,
    epsilon_decay = 0, epsilon_min = 0.01, action_selection_method = "e-greedy", temperature = None,
    func_approx_type = 'fourier', order = 1, num_state_dimensions = 5, num_states = ec.STATE_NUM)


    def reset(self):
        self.ac.reset()

    def choose_action(self, state_results):
        return self.ac.select_action(state_results)

    def learn(self, state_results, state_results_next, action_idx):
        reward = ec.reward_function(state_results,state_results_next)
        td_error = self.ac.get_value_TD_error(reward, state_results, state_results_next)
        self.ac.update_critic(state_results, action_idx, td_error)
        self.ac.update_actor(state_results, action_idx, td_error)
