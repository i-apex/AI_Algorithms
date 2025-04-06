# record_idastar.py

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from frozen_lake import ida_star

env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")
env = RecordVideo(env, video_folder="./videos", name_prefix="idastar_run", episode_trigger=lambda i: i == 0)


P = env.unwrapped.P

# start_state comes from reset()
obs, _ = env.reset()
start_state = obs
desc = env.unwrapped.desc
goal_state = None
for s in range(env.observation_space.n):
    if desc.flat[s] == b'G':
        goal_state = s
        break
path, cost = ida_star(P, start_state, goal_state, env.action_space.n)

if path is not None:
    print(f"Found path: {path}, Cost: {cost}")
    for action in path:
        state, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
else:
    print("No path found")

env.close()
