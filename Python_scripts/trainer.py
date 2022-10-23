from mlagents_envs.environment import UnityEnvironment
import numpy as np
from experience import Experience, Trajectory, Buffer
from cnn import CNN
from agent import Agent


class Trainer:
    def __init__(self) -> None:
        pass

    @staticmethod  # Static methods don't take the argument self
    def generate_trajectories(env: UnityEnvironment, agent: Agent, sample_size: int, epsilon: float):  # Generates data to train with
        # Pass entire agent to function to have its id, number etc. as well
        # Colab example adds epsilon noise to the trajectories. I suppose that this promotes exploration, therefore I kept it as well.

        def reset_environment():  # This might create inconsistencies in the environment interface when having multiple agent groups, but I don't care for now
            env.reset()
            agent.increment_agent_num()

        def get_next_experience():
            obs = env.get_step_result(agent.group).get_agent_step_result(agent.num).obs[0]  # Get observation. Not formatted as tensor!
            action = agent.get_action(obs)  # Get agent actions
            # TODO: Add noise on agent actions. CAREFUL: respect constraints of action ranges!
            env.set_actions(agent.group, action)  # Set actions
            env.step()
            next_step_res = env.get_step_result(agent.group)
            reward = next_step_res.reward  # Get reward
            done = next_step_res.done[agent.index]  # Get done flag
            next_obs = next_step_res.get_agent_step_result(agent.num).obs[0]  # Get next observation

            return Experience(obs=obs, action=action, reward=reward, done=done, next_obs=next_obs)

        buffer: Buffer = []
        rewards = []
        reset_environment()

        while len(buffer) < sample_size:
            trajectory: Trajectory = []
            trajectory_finished = False
            reward = 0
            while not trajectory_finished:
                next_exp = get_next_experience()
                trajectory.append(next_exp)
                trajectory_finished = next_exp.done
                reward += next_exp.reward
            buffer.extend(trajectory)
            rewards.append(reward)
            reset_environment()

        return buffer, np.sum(rewards) / len(buffer)
    
    @staticmethod
    def update_q_network():  # Training happens here
        raise NotImplementedError