from gym_unity.envs import UnityEnv, logger, UnityGymException, ActionFlattener, AgentIdIndexMapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import BatchedStepResult
import numpy as np
from typing import Set
from gym import spaces


class GymUnityWrapper(UnityEnv):  # Subclass UnityEnv to include support for side channels
    def __init__(  # Completely copy-pasting UnityEnv __init__() in order to include the support for side channels without creating env twice
        self,
        environment_filename: str,
        worker_id: int = 0,
        use_visual: bool = False,
        uint8_visual: bool = False,
        multiagent: bool = False,
        flatten_branched: bool = False,
        no_graphics: bool = False,
        allow_multiple_visual_obs: bool = False,
        side_channels: list = [],  # Added
    ):
        """
        Environment initialization
        :param environment_filename: The UnityEnvironment path or file to be wrapped in the gym.
        :param worker_id: Worker number for environment.
        :param use_visual: Whether to use visual observation or vector observation.
        :param uint8_visual: Return visual observations as uint8 (0-255) matrices instead of float (0.0-1.0).
        :param multiagent: Whether to run in multi-agent mode (lists of obs, reward, done).
        :param flatten_branched: If True, turn branched discrete action spaces into a Discrete space rather than
            MultiDiscrete.
        :param no_graphics: Whether to run the Unity simulator in no-graphics mode
        :param allow_multiple_visual_obs: If True, return a list of visual observations instead of only one.
        """
        base_port = 5005
        if environment_filename is None:
            base_port = UnityEnvironment.DEFAULT_EDITOR_PORT

        self._env = UnityEnvironment(
            environment_filename,
            worker_id,
            base_port=base_port,
            no_graphics=no_graphics,
            side_channels=side_channels,  # Added
        )

        # Take a single step so that the brain information will be sent over
        if not self._env.get_agent_groups():
            self._env.step()

        self.visual_obs = None
        self._n_agents = -1

        self.agent_mapper = AgentIdIndexMapper()

        # Save the step result from the last time all Agents requested decisions.
        self._previous_step_result: BatchedStepResult = None
        self._multiagent = multiagent
        self._flattener = None
        # Hidden flag used by Atari environments to determine if the game is over
        self.game_over = False
        self._allow_multiple_visual_obs = allow_multiple_visual_obs

        # Check brain configuration
        if len(self._env.get_agent_groups()) != 1:
            raise UnityGymException(
                "There can only be one brain in a UnityEnvironment "
                "if it is wrapped in a gym."
            )

        self.brain_name = self._env.get_agent_groups()[0]
        self.name = self.brain_name
        self.group_spec = self._env.get_agent_group_spec(self.brain_name)

        if use_visual and self._get_n_vis_obs() == 0:
            raise UnityGymException(
                "`use_visual` was set to True, however there are no"
                " visual observations as part of this environment."
            )
        self.use_visual = self._get_n_vis_obs() >= 1 and use_visual

        if not use_visual and uint8_visual:
            logger.warning(
                "`uint8_visual was set to true, but visual observations are not in use. "
                "This setting will not have any effect."
            )
        else:
            self.uint8_visual = uint8_visual

        if self._get_n_vis_obs() > 1 and not self._allow_multiple_visual_obs:
            logger.warning(
                "The environment contains more than one visual observation. "
                "You must define allow_multiple_visual_obs=True to received them all. "
                "Otherwise, please note that only the first will be provided in the observation."
            )

        # Check for number of agents in scene.
        self._env.reset()
        step_result = self._env.get_step_result(self.brain_name)
        self._check_agents(step_result.n_agents())
        self._previous_step_result = step_result
        self.agent_mapper.set_initial_agents(list(self._previous_step_result.agent_id))

        # Set observation and action spaces
        if self.group_spec.is_action_discrete():
            branches = self.group_spec.discrete_action_branches
            if self.group_spec.action_shape == 1:
                self._action_space = spaces.Discrete(branches[0])
            else:
                if flatten_branched:
                    self._flattener = ActionFlattener(branches)
                    self._action_space = self._flattener.action_space
                else:
                    self._action_space = spaces.MultiDiscrete(branches)

        else:
            if flatten_branched:
                logger.warning(
                    "The environment has a non-discrete action space. It will "
                    "not be flattened."
                )
            high = np.array([1] * self.group_spec.action_shape)
            self._action_space = spaces.Box(-high, high, dtype=np.float32)
        high = np.array([np.inf] * self._get_vec_obs_size())
        if self.use_visual:
            shape = self._get_vis_obs_shape()
            if uint8_visual:
                self._observation_space = spaces.Box(
                    0, 255, dtype=np.uint8, shape=shape
                )
            else:
                self._observation_space = spaces.Box(
                    0, 1, dtype=np.float32, shape=shape
                )

        else:
            self._observation_space = spaces.Box(-high, high, dtype=np.float32)