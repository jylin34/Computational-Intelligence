class Agent:
    def __init__(self, lr, discount, epsilon, epsilon_decay, min_epsilon):
        self.q_table = {}
        self.lr = lr
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def get_state(self, sensor_values):
        
    def select_action(self, state):

    def update_q_table(self, state, action, reward, next_state):
        
