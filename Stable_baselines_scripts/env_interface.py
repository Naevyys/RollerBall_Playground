from gym_unity.envs import UnityEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.evaluation import evaluate_policy
#from stable_baselines3.common.env_checker import check_env
import os, pickle
import numpy as np
from models import CNN
import matplotlib.pyplot as plt


def make_env(path_to_exec:str, log_dir:str="./logs/", worker_id:int=0, no_graphics:bool=True):
    os.makedirs(log_dir, exist_ok=True)  # Create log directory if not already existing
    env = UnityEnv(path_to_exec, worker_id=worker_id, use_visual=True, uint8_visual=True, no_graphics=no_graphics)  # no_graphics=True to avoid popping the unity view
    env = Monitor(env, log_dir, allow_early_resets=True)
    #check_env(env)  # Check wheather my unity environment follows the Gym API. See https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html
    # This check_env function always interrupts my code because it says my reward is not of type float, but I checked manually and the reward is a numpy.float32, which indeed does not pass the assertion isinstance(reward, (int, float)). However, since this assertion fails, I don't know if there are any further issues with my custom environment.
    return env

def make_model(env: UnityEnv, model_class:BaseAlgorithm, policy:BasePolicy, init_params:dict()):
    #policy_kwargs = dict(  # Can also be passed in init_params instead of doing this here, see that later
    #    features_extractor_class=CNN,
    #    features_extractor_kwargs=dict(features_dim=32)
    #)  # Use a custom CNN instead of the NatureCNN of stable_baseline3
    return model_class(policy, env, **init_params)#, policy_kwargs=policy_kwargs)

def train_model(model:BaseAlgorithm, train_params, model_dir:str):
    # Note that the library is directly taking the observations and pre-processing them following standard procedures for images (e.g. standardize)
    try:
        model.learn(**train_params)
        model.save(model_dir)
    except KeyboardInterrupt:  # Not tested, supposed to stop training using keyboard interrupt and save the current model
        model.save(model_dir)

def evaluate_model(model:BaseAlgorithm, eval_dir:str, episodes:int = 50):
    episode_rewards, episode_lengths = evaluate_policy(model, model.get_env(), n_eval_episodes=episodes, deterministic=False, return_episode_rewards=True)
    print("Mean reward: {:0.3f} \t mean length: {:0.3f}".format(np.mean(episode_rewards), np.mean(episode_lengths)))
    os.makedirs(eval_dir, exist_ok=True)
    with open('{}/eval.pkl'.format(eval_dir), 'wb') as f:
        pickle.dump(episode_rewards, f)
        pickle.dump(episode_lengths, f)

def load_model(path:str, env:UnityEnv, model_class:BaseAlgorithm):
    return model_class.load(path, env=env)

def run_model(env:UnityEnv, model:BaseAlgorithm, n_steps:int=1000, deterministic=False):
    obs = env.reset()
    for i in range(n_steps):
        action, _states = model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, info = env.step(action)
        env.render()
        if dones:
            obs = env.reset()