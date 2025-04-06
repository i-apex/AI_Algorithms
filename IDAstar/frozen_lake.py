import gymnasium as gym
import time
import matplotlib.pyplot as plt
from collections import deque
from gymnasium.wrappers import RecordVideo


class Node:
    def __init__(self, state, path, cost):
        self.state = state
        self.path = path
        self.cost = cost


def ida_star(P, start, goal_state, action_space_n):
    def h(state):
        row_s, col_s = divmod(state, 4)
        row_g, col_g = divmod(goal_state, 4)
        return abs(row_s - row_g) + abs(col_s - col_g)

    def dfs(path, g, bound, visited):
        node = path[-1]
        f = g + h(node.state)
        if f > bound:
            return f, None
        if node.state == goal_state:
            return -1, node.path
        min_cost = float('inf')
        for action in range(action_space_n):
            for prob, next_state, reward, done in P[node.state][action]:
                if prob > 0 and next_state not in visited:
                    visited.add(next_state)
                    new_node = Node(next_state, node.path + [action], g + 1)
                    path.append(new_node)
                    t, found = dfs(path, g + 1, bound, visited)
                    if found is not None:
                        return -1, found
                    if t < min_cost:
                        min_cost = t
                    path.pop()
                    visited.remove(next_state)
        return min_cost, None

    bound = h(start)
    path = [Node(start, [], 0)]
    visited = set([start])
    while True:
        t, result = dfs(path, 0, bound, visited)
        if result is not None:
            return result, len(result)
        if t == float('inf'):
            return None, 0
        bound = t


# ✅ Create and wrap environment with video recording
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")
env = RecordVideo(env, video_folder="./video", name_prefix="ida_star", episode_trigger=lambda x: True)

# ✅ Reset and get the initial state
obs, _ = env.reset()
start_state = obs
P = env.unwrapped.P
desc = env.unwrapped.desc

# ✅ Locate goal state
goal_state = None
for s in range(env.observation_space.n):
    if desc.flat[s] == b'G':
        goal_state = s
        break

# ✅ Run IDA* from start to goal
start_time = time.time()
path, cost = ida_star(P, start_state, goal_state, env.action_space.n)
end_time = time.time()

print(f"Found path: {path}, Cost: {cost}")
print(f"Execution Time: {end_time - start_time:.4f} sec")

# ✅ Replay the found path so it's recorded by RecordVideo
obs, _ = env.reset()
for action in path:
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        break

env.close()

# ✅ Plot time taken
plt.plot([end_time - start_time], label="Time Taken")
plt.xlabel("Run")
plt.ylabel("Time (s)")
plt.title("IDA* on Frozen Lake")
plt.legend()
plt.savefig("ida_star_frozenlake.png")
plt.show()
