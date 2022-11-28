from stable_baselines3.common.callbacks import BaseCallback

class RolloutCheckCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RolloutCheckCallback, self).__init__(verbose)
        #self.actions = []
        #self.clipped_actions = []
        #self.obs_min = []
        #self.obs_max = []
        self.obs = []
        self.rollout_buffers = []
        self.rewards = []
        self.dones = []
        #self.total_timesteps_of_model = None
        #self.n_steps_of_model = None
        #self.n_rollout_steps = None
        #self.n_steps_vals_at_rollout_end = []
        #self.num_timesteps_at_training_start = None
        #self.num_timesteps_at_training_end = None
        #self.iteration_at_training_end = None
        #self.num_envs = None
    
    #def _on_training_start(self):
        #self.total_timesteps_of_model = self.locals["total_timesteps"]
        #self.n_steps_of_model = self.model.n_steps
        #self.num_timesteps_at_training_start = self.model.num_timesteps
        #self.num_envs = self.model.env.num_envs

    def _on_step(self):
        #self.actions.append(self.locals["actions"])
        #self.clipped_actions.append(self.locals["clipped_actions"])
        #self.obs_min.append(self.locals["new_obs"].min())
        #self.obs_max.append(self.locals["new_obs"].max())
        self.obs.append(self.locals["new_obs"])
        self.rewards.append(self.locals["rewards"])
        self.dones.append(self.locals["dones"])
        #self.n_rollout_steps = self.locals["n_rollout_steps"]
        return True
    
    def _on_rollout_end(self):
        #self.n_steps_vals_at_rollout_end.append(self.locals["n_steps"])
        self.rollout_buffers.append(self.locals["rollout_buffer"])
    
    #def _on_training_end(self):
        #self.num_timesteps_at_training_end = self.model.num_timesteps
        #self.iteration_at_training_end = self.locals["iteration"]  # Locals are not updated before calling this method, so this info is not reliable
    


# gym-unity how to specify observation space