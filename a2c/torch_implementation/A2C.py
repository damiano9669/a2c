import numpy as np
import torch
import torch.optim as optim

from a2c.basic_agent.A2CBasicAgent import A2CBasicAgent


class A2C(A2CBasicAgent):

    def __init__(self,
                 actor_critic_neural_network,
                 learning_rate=3e-4,
                 gamma=0.99,
                 model_path='models/model.pt',
                 chart_path='charts/plot.png'):
        super(A2C, self).__init__(learning_rate=learning_rate,
                                  gamma=gamma,
                                  model_path=model_path,
                                  chart_path=chart_path)

        self.actor_critic = actor_critic_neural_network
        self.ac_optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate)

    def get_action(self, state, training=True):
        value, policy_dist = self.actor_critic.forward(state)
        value = value.detach().numpy()[0, 0]
        dist = policy_dist.detach().numpy()

        if training:
            action = np.random.choice(dist.shape[1], p=np.squeeze(dist))
            log_prob = torch.log(policy_dist.squeeze(0)[action])
            entropy = -np.sum(np.mean(dist) * np.log(dist))

            self.update_values_logprobs_entropy(value, log_prob, entropy)
        else:
            action = np.argmax(dist)

        return action

    def update_model(self, state):
        Qval, _ = self.actor_critic.forward(state)
        Qval = Qval.detach().numpy()[0, 0]
        self.all_rewards.append(np.sum(self.rewards))

        # compute Q values
        Qvals = np.zeros_like(self.values)
        for t in reversed(range(len(self.rewards))):
            Qval = self.rewards[t] + self.gamma * Qval
            Qvals[t] = Qval

        # update actor critic
        values = torch.FloatTensor(self.values)
        Qvals = torch.FloatTensor(Qvals)
        log_probs = torch.stack(self.log_probs)

        advantage = Qvals - values
        actor_loss = (-log_probs * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss + 0.001 * self.entropy_term

        self.ac_optimizer.zero_grad()
        ac_loss.backward()
        self.ac_optimizer.step()

        # reset
        self.reset_values_logprobs_rewards()

    def save_model(self):
        super(A2C, self).save_model()
        torch.save(self.actor_critic.state_dict(), self.model_path)

    def load_model(self):
        super(A2C, self).load_model()
        self.actor_critic.load_state_dict(torch.load(self.model_path))
