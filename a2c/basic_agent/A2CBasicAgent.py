import pandas as pd

from a2c.plot_style.cb91visuals import *


class A2CBasicAgent:

    def __init__(self,
                 learning_rate=3e-4,
                 gamma=0.99,
                 model_path='models/model',
                 chart_path='charts/plot.png'):
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.model_path = model_path
        self.chart_path = chart_path

        self.all_rewards = []
        self.entropy_term = 0

        self.log_probs = []
        self.values = []
        self.rewards = []

    def get_action(self, state, training=True):
        pass

    def update_model(self, state):
        pass

    def update_values_logprobs_entropy(self, value, log_prob, entropy):
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.entropy_term += entropy

    def update_reward(self, reward):
        self.rewards.append(reward)

    def reset_values_logprobs_rewards(self):
        self.values = []
        self.log_probs = []
        self.rewards = []

    def save_model(self):
        print(f'Saving model to path: {self.model_path}')
        pass

    def load_model(self):
        print(f'Loading model from path: {self.model_path}')
        pass

    def plot_rewards(self):
        smoothed_rewards = pd.Series.rolling(pd.Series(self.all_rewards), 10).mean()
        smoothed_rewards = [elem for elem in smoothed_rewards]
        plt.plot(self.all_rewards, alpha=0.3, c='C0')
        plt.plot(smoothed_rewards, c='C0')
        plt.plot()
        plt.xlabel('Episode')
        plt.ylabel('Reward')

        plt.savefig(self.chart_path)
        plt.show()
