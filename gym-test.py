import gymnasium as gym
import numpy as np
import time
from minigrid.wrappers import FullyObsWrapper
# env = gym.make("MiniGrid-NineRoomsKeyGoal-v0", render_mode="human")
env = gym.make("MiniGrid-NineRoomsKeyGoal-v0")

observation, info = env.reset(seed=42)
print("observation: ", observation['image'].shape)
env = FullyObsWrapper(env)
observation, info = env.reset(seed=42)
print("observation: ", observation['image'].shape)
observation, info = env.reset(seed=42)
for _ in range(1000):
   action = np.random.choice(env.action_space.n)  # User-defined policy function
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()
time.sleep(100)
env.close()