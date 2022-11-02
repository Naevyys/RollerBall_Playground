from gym_unity.envs import UnityEnv, spaces
from stable_baselines3 import A2C
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
import os, pickle
import numpy as np


def make_env(path_to_exec:str, log_dir:str="./logs/", worker_id:int=0, no_graphics:bool=True):
    os.makedirs(log_dir, exist_ok=True)  # Create log directory if not already existing
    env = UnityEnv(path_to_exec, worker_id=worker_id, use_visual=True, no_graphics=no_graphics)  # no_graphics=True to avoid popping the unity view
    env = Monitor(env, log_dir, allow_early_resets=True)
    env.observation_space = spaces.Box(low=0, high=255, shape=(36, 36, 1), dtype=np.uint8)
    #check_env(env)  # Check wheather my unity environment follows the Gym API. See https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html
    # This check_env function always interrupts my code because it says my reward is not of type float, but my code is working and I checked my C# code and it always rewards with floats, so I don't know where the issue comes from, but I'm forced to comment this function out to be able to run my code...
    return env

def make_model(env: UnityEnv, verbose:int=1):
    return A2C(ActorCriticCnnPolicy, env, verbose=verbose)

def train_model(model:A2C, env:UnityEnv, model_dir:str="./models/", log_dir:str="./logs/", episodes:int=3, timesteps:int=100):
    # Note that the library is directly taking the observations and pre-processing them following standard procedures for images (e.g. standardize)
    model.learn(total_timesteps=timesteps)
    model.save(model_dir)
    episode_rewards = []
    episode_lengths = []
    
    for _ in range(episodes):
        obs = env.reset()
        total_reward = 0
        total_length = 0
        while True:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            total_length += 1
            if done:
                break
        episode_rewards.append(total_reward)
        episode_lengths.append(total_length)
    
    mean_reward = np.mean(episode_rewards)
    mean_length = np.mean(episode_lengths)
    print("Episode mean reward: {:0.3f} \t mean length: {:0.3f}".format(mean_reward, mean_length))

    with open('{}_eval.pkl'.format(log_dir), 'wb') as f:
        pickle.dump(episode_rewards, f)
        pickle.dump(episode_lengths, f)
