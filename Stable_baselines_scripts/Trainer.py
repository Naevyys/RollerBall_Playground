#from gym_unity.envs import UnityEnv
#from mlagents_envs.environment import UnityEnvironment
from GymUnityWrapper import GymUnityWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.evaluation import evaluate_policy
from datetime import datetime
import os, pickle
import numpy as np

class Trainer(object):
    def __init__(self, name:str, env_exec:str, models_folder:str, model_class:BaseAlgorithm, policy_class:BasePolicy, model_init_params:dict(), worker_id:int=0, no_graphics:bool=True, run_id:int=None, side_channels:list=[]) -> None:
        self.name = name
        self.env_exec = env_exec
        self.models_folder = models_folder
        self.model_class = model_class
        self.side_channels = side_channels

        # Init paths
        self.run_id = datetime.now().strftime("%d%m%Y_%H%M%S") if run_id is None else run_id
        self.base_path = "./{}/{}/{}_{}/"
        self.training_log_dir = self.base_path.format(models_folder, "training_logs", self.name, self.run_id)
        self.eval_log_dir = self.base_path.format(models_folder, "eval_logs", self.name, self.run_id)
        self.env_log_dir = self.base_path.format(models_folder, "logs", self.name, self.run_id)
        self.model_dir = self.base_path.format(models_folder, "models", self.name, self.run_id)

        # Init logging dirs
        os.makedirs(self.eval_log_dir, exist_ok=True)
        os.makedirs(self.env_log_dir, exist_ok=True)
        # Other logdirs will be created automatically by stable baselines3

        # Init env
        env = GymUnityWrapper(env_exec, worker_id=worker_id, use_visual=True, uint8_visual=True, no_graphics=no_graphics, side_channels=self.side_channels)
        #env = UnityEnv(env_exec, worker_id=worker_id, use_visual=True, uint8_visual=True, no_graphics=no_graphics)  # no_graphics=True to avoid popping the unity view
        #env._env.close()  # Close previous environment
        #env._env = UnityEnvironment(env_exec, worker_id, base_port=5005, no_graphics=no_graphics, side_channels=self.side_channels)  # Manually create a new unity environment inside the gym wrapper in order to include the side channels, not nice, but this workaround is needed in version 0.14.1 cause the official update only comes in later versions. See https://forum.unity.com/threads/sidechannel-in-gym-environment.863938/ for more details, I do it here instead of inside the library cause I don't want to modify library files
        self.env = Monitor(env, self.env_log_dir, allow_early_resets=True)
        # check_env(self.env)  # Check wheather my unity environment follows the Gym API. See https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html (*)

        # Init model
        self.model = self.model_class(policy_class, env, tensorboard_log=self.training_log_dir, **model_init_params)

    def train(self, train_params:dict()):  # Train model
        try:
            self.model.learn(**train_params)
            self.model.save(self.model_dir)
        except KeyboardInterrupt:  # Not tested, supposed to stop training using keyboard interrupt and save the current model
            self.model.save(self.model_dir)
    
    def load(self):  # Load model
        self.model = self.model_class.load(self.model_dir, env=self.env)
    
    def evaluate(self, episodes:int=10, deterministic:bool=False):
        episode_rewards, episode_lengths = evaluate_policy(self.model, self.model.get_env(), n_eval_episodes=episodes, deterministic=deterministic, return_episode_rewards=True)
        print("Mean reward: {:0.3f} \t mean length: {:0.3f}".format(np.mean(episode_rewards), np.mean(episode_lengths)))
        with open('{}/eval.pkl'.format(self.eval_log_dir), 'wb') as f:
            pickle.dump(episode_rewards, f)
            pickle.dump(episode_lengths, f)
    
    def run(self, n_steps:int=100, deterministic:bool=False):
        obs = self.env.reset()
        for _ in range(n_steps):
            action, _states = self.model.predict(obs, deterministic=deterministic)
            obs, rewards, dones, info = self.env.step(action)
            if len(self.side_channels) > 0:
                agent_pos_x, agent_pos_z = self.side_channels[0].get_property("agent_position_x"), self.side_channels[0].get_property("agent_position_z")
                print(agent_pos_x, agent_pos_z)
            self.env.render()
            if dones:
                obs = self.env.reset()
    
    def close_env(self):
        self.env.close()


# (*) This check_env function always interrupts my code because it says my reward is not of type float, but I checked manually and the reward is a numpy.float32, which indeed does not pass the assertion isinstance(reward, (int, float)). However, since this assertion fails, I don't know if there are any further issues with my custom environment.