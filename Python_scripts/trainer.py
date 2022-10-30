from mlagents_envs.environment import UnityEnvironment
import numpy as np
import torch
from experience import Experience, Trajectory, Buffer
from models import ActorCritic, ActorCriticLoss
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
            action += np.random.randn(action.shape[0], action.shape[1]).astype(np.float32) * epsilon  # Add some random value to encourage exploration
            env.set_actions(agent.group, action)  # Set actions
            env.step()
            next_step_res = env.get_step_result(agent.group)
            reward = next_step_res.reward[0]  # Get reward
            done = next_step_res.done[agent.index]  # Get done flag
            next_obs = next_step_res.get_agent_step_result(agent.num).obs[0]  # Get next observation

            return Experience(obs=obs, action=action, reward=reward, done=done, next_obs=next_obs)

        buffer: Buffer = []
        rewards = []
        env.reset()

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
    def update_network(net: ActorCritic, optimizer: torch.optim, buffer: Buffer, n_epochs: int, batch_size: int, gamma: float, criterion: ActorCriticLoss):  # Training happens here

        np.random.shuffle(buffer)  # Shuffle experiments in buffer
        batches = [buffer[batch_size * start: batch_size * (start + 1)] for start in range(int(len(buffer) / batch_size))]  # Partition in batches
        
        for i in range(n_epochs):
            for j, batch in enumerate(batches):
                # Create batch tensors for observations, reward, done, action and next observations
                obs = torch.from_numpy(np.stack([ex.obs for ex in batch])).permute(0, 3, 1, 2)  # Move channel dimension to second dim
                reward = torch.from_numpy(np.array([ex.reward for ex in batch], dtype=np.float32).reshape(-1, 1))
                done = torch.from_numpy(np.array([ex.done for ex in batch], dtype=np.float32).reshape(-1, 1))
                action = torch.from_numpy(np.stack([ex.action for ex in batch])).squeeze(dim=1)  # Remove dimension 1
                next_obs = torch.from_numpy(np.stack([ex.next_obs for ex in batch])).permute(0, 3, 1, 2)  # Move channel dimension

                # Following https://medium.com/@asteinbach/rl-introduction-simple-actor-critic-for-continuous-actions-4e22afb712
                means_vec, vars_vec, td_vec = net(obs)
                _, _, next_values = net(next_obs)
                expected_td_vec = reward + gamma * next_values * (1 - done)

                # Compute loss
                loss = criterion(means_vec, vars_vec, action, td_vec,expected_td_vec)

                # Perform the backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print("Batch {} done".format(j))
            print("Epoch {} done".format(i))
    
    def save_network(net: ActorCritic, path: str):
        torch.save(net.state_dict(), path)


# Notes for later:
# - Standardize sequence of returns to stabilize training (https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic section 2 of training)