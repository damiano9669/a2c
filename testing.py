from pathlib import Path

import gym
import numpy as np
import tensorflow as tf

from a2c.tensorflow_implementation.A2C import A2C as A2CTensorflow
from a2c.tensorflow_implementation.ActorCriticNetwork import ActorCriticNetwork as ActorCriticNetworkTensorflow
from a2c.torch_implementation.A2C import A2C as A2CTorch
from a2c.torch_implementation.ActorCriticNetwork import ActorCriticNetwork as ActorCriticNetworkTorch


def test(a2c_algorithm, environment, n_episodes=3000, load_model=False, training=True):
    if load_model:
        a2c_algorithm.load_model()

    best_reward = 0
    for episode in range(n_episodes):

        episode_total_reward = 0
        observation = environment.reset()

        done = False
        while not done:
            action = a2c_algorithm.get_action(observation, training=training)

            observation, reward, done, _ = environment.step(action)
            episode_total_reward += reward

            a2c_algorithm.update_reward(reward)

        if training:
            a2c_algorithm.update_model(observation)

        if episode_total_reward > best_reward:
            print(f'Episode: {episode} - reward: {episode_total_reward}')
            best_reward = episode_total_reward
            a2c_algorithm.save_model()
            a2c_algorithm.plot_rewards()
            print('-' * 60)

        elif episode % 100 == 0:
            a2c_algorithm.plot_rewards()
            print(f'Episode: {episode} - reward: {episode_total_reward}')
            print('-' * 60)

    a2c_algorithm.plot_rewards()


if __name__ == '__main__':
    env_name = 'CartPole-v0'
    env = gym.make(env_name)

    print(f'Environment name: {env_name}\n'
          f'Action space: {env.action_space}\n'
          f'Observation space: {env.observation_space}')
    print('-' * 60)

    print('Testing PyTorch implementation:')
    actor_critic_nn = ActorCriticNetworkTorch(input_size=env.observation_space.shape[0],
                                              number_actions=env.action_space.n)
    a2c = A2CTorch(actor_critic_neural_network=actor_critic_nn,
                   model_path=Path(__file__).parent / 'models/torch_model.pt',
                   chart_path=Path(__file__).parent / 'charts/torch.png')
    # run test
    test(a2c, env, n_episodes=1500, load_model=False, training=True)
    # inference test
    test(a2c, env, n_episodes=1, load_model=True, training=False)
    print('-' * 60)

    print('Testing Tensorflow implementation:')
    actor_critic_nn = ActorCriticNetworkTensorflow(input_size=env.observation_space.shape[0],
                                                   number_actions=env.action_space.n)
    actor_critic_nn(tf.convert_to_tensor(np.random.random((1, env.observation_space.shape[0])), dtype=tf.float32))
    a2c = A2CTensorflow(actor_critic_neural_network=actor_critic_nn,
                        model_path=Path(__file__).parent / 'models/tensorflow_model.pt',
                        chart_path=Path(__file__).parent / 'charts/tensorflow.png')
    # run test
    test(a2c, env, n_episodes=1500, load_model=False, training=True)
    # inference test
    test(a2c, env, n_episodes=1, load_model=True, training=False)
    print('-' * 60)
