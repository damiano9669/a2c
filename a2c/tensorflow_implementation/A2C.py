import numpy as np
import tensorflow as tf

from a2c.basic_agent.A2CBasicAgent import A2CBasicAgent


class A2C(A2CBasicAgent):

    def __init__(self,
                 actor_critic_neural_network,
                 learning_rate=3e-4,
                 gamma=0.99,
                 model_path='models/model.h5',
                 chart_path='charts/plot.png'):
        super(A2C, self).__init__(learning_rate=learning_rate,
                                  gamma=gamma,
                                  model_path=model_path,
                                  chart_path=chart_path)

        self.actor_critic = actor_critic_neural_network
        self.ac_optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)

    def get_action(self, state, training=True):
        value, policy_dist = self.actor_critic(tf.convert_to_tensor(state[None, :], dtype=tf.float32))
        value = value.numpy()[0, 0]
        dist = policy_dist.numpy()[0]

        if training:
            action = np.random.choice(dist.shape[0], p=dist)
        else:
            action = np.argmax(dist)

        log_prob = np.log(dist[action])
        entropy = -np.sum(np.mean(dist) * np.log(dist))

        self.update_values_logprobs_entropy(value, log_prob, entropy)

        return action

    def update_model(self, state):
        with tf.GradientTape() as tape:
            Qval, _ = self.actor_critic(tf.convert_to_tensor(state[None, :], dtype=tf.float32))
            Qval = Qval.numpy()[0, 0]
            self.all_rewards.append(np.sum(self.rewards))

            # compute Q values
            Qvals = np.zeros_like(self.values)
            for t in reversed(range(len(self.rewards))):
                Qval = self.rewards[t] + self.gamma * Qval
                Qvals[t] = Qval

            # update actor critic
            values = tf.convert_to_tensor(self.values, dtype=tf.float32)
            Qvals = tf.convert_to_tensor(Qvals, dtype=tf.float32)
            log_probs = tf.convert_to_tensor(np.stack(self.log_probs, axis=0), dtype=tf.float32)

            advantage = Qvals - values
            actor_loss = tf.reduce_mean((-log_probs * advantage))
            critic_loss = 0.5 * tf.reduce_mean((tf.pow(advantage, 2)))
            ac_loss = actor_loss + critic_loss + 0.001 * self.entropy_term

        grads = tape.gradient(ac_loss, self.actor_critic.trainable_weights)
        self.ac_optimizer.apply_gradients(zip(grads, self.actor_critic.trainable_weights))

        # reset
        self.reset_values_logprobs_rewards()

    def save_model(self):
        super(A2C, self).save_model()
        self.actor_critic.save_weights(self.model_path)

    def load_model(self):
        super(A2C, self).load_model()
        self.actor_critic.load_weights(self.model_path)
