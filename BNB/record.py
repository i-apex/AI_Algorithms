import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from frozen_lake import branch_and_bound  # Assuming your BnB logic is here

# Create environment with 'rgb_array' rendering for video
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")

# Wrap with RecordVideo
env = RecordVideo(env, video_folder="./videos", name_prefix="bnb_run", episode_trigger=lambda x: True)

obs, _ = env.reset()
path, _ = branch_and_bound(env)

if path:
    for action in path:
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
env.close()
