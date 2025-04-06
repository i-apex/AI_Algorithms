import gymnasium as gym
import time
import matplotlib.pyplot as plt
from queue import PriorityQueue



class Node:
    def __init__(self, state, path, cost):
        self.state = state
        self.path = path
        self.cost = cost

    def __lt__(self, other):
        return self.cost < other.cost

def branch_and_bound(env):
    pq = PriorityQueue()
    pq.put(Node(env.reset()[0], [], 0))
    visited = set()

    P = env.unwrapped.P  # Transition dynamics

    while not pq.empty():
        node = pq.get()
        state, path, cost = node.state, node.path, node.cost

        if state in visited:
            continue
        visited.add(state)

        if env.unwrapped.desc.flat[state] == b'G':
            return path, cost

        for action in range(env.action_space.n):
            transitions = P[state][action]
            for prob, next_state, reward, done in transitions:
                if prob > 0 and next_state not in visited:
                    pq.put(Node(next_state, path + [action], cost + 1))
    return None, 0



env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="ansi")
rewards = []
times = []

for i in range(5):
    env.reset()
    path, exec_time = branch_and_bound(env)
    print(f"Run {i+1}: Path = {path}, Time = {exec_time:.4f} sec")
    rewards.append(1 if path else 0)
    times.append(exec_time)

print(f"\nAverage Reward: {sum(rewards)/5}")
print(f"Average Time: {sum(times)/5:.4f} sec")


plt.plot(times, label="Time Taken per Run")
plt.xlabel("Run")
plt.ylabel("Time (s)")
plt.title("Branch and Bound on Frozen Lake")
plt.legend()
plt.savefig("bnb_frozenlake.png")
plt.show()
