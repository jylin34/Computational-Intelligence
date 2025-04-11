import random
import math

class Agent:
    def __init__(self, lr, discount_factor, epsilon, epsilon_decay, min_epsilon=0.05, num_actions=5):
        self.q_table = {}
        self.lr = lr
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.num_actions = num_actions

    def get_state(self, sensor_values):
        bins = [int(min(s // 5, 9)) for s in sensor_values] # 5~10 10~15, 5 for each segment
        return tuple(bins)

    def select_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0.0] * self.num_actions

        # Epsilon-Greedy
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            return int(self.argmax(self.q_table[state])) 

    def update_q_table(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = [0.0] * self.num_actions
        if next_state not in self.q_table:
            self.q_table[next_state] = [0.0] * self.num_actions

        expected_q = self.q_table[state][action] # expected total reward
        observed_q = reward + self.gamma *  max(self.q_table[next_state])
        TD_error = observed_q - expected_q
        updated_q = expected_q + self.lr * TD_error
        self.q_table[state][action] = updated_q

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    @staticmethod
    def argmax(lst):
        max_val = max(lst)
        indices = [i for i, val in enumerate(lst) if val == max_val]
        return random.choice(indices)
