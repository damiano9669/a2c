import tensorflow as tf


class ActorCriticNetwork(tf.keras.Model):
    def __init__(self, input_size, number_actions):
        super(ActorCriticNetwork, self).__init__()

        hidden_size = 256

        self.critic_linear1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.critic_linear2 = tf.keras.layers.Dense(1)

        self.actor_linear1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.actor_linear2 = tf.keras.layers.Dense(number_actions, activation='softmax')

    def call(self, state):
        value = self.critic_linear1(state)
        value = self.critic_linear2(value)

        policy_dist = self.actor_linear1(state)
        policy_dist = self.actor_linear2(policy_dist)

        return value, policy_dist
