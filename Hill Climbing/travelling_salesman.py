import numpy as np
import matplotlib.pyplot as plt
import imageio
from TSP_env import TSPEnv
import os

# Parameters
num_nodes = 20
env = TSPEnv(num_nodes=num_nodes, batch_size=1, num_draw=1)
state = env.reset()[0]  # Single environment
coords = state[:, :2]

# Distance calculation
def total_distance(route):
    dist = 0
    for i in range(len(route)):
        dist += np.linalg.norm(coords[route[i]] - coords[route[(i + 1) % len(route)]])
    return dist

# Hill Climbing
def hill_climbing(init_route, iterations=1000):
    current_route = init_route.copy()
    current_cost = total_distance(current_route)
    best_route = current_route.copy()
    best_cost = current_cost
    images = []
    costs = []

    os.makedirs("frames", exist_ok=True)

    for i in range(iterations):
        a, b = np.random.choice(len(current_route), size=2, replace=False)
        new_route = current_route.copy()
        new_route[a], new_route[b] = new_route[b], new_route[a]

        new_cost = total_distance(new_route)

        if new_cost < current_cost:
            current_route = new_route
            current_cost = new_cost

            if new_cost < best_cost:
                best_route = new_route
                best_cost = new_cost

        # Save frame and cost every 50 steps
        if i % 50 == 0 or i == iterations - 1:
            fig, ax = plt.subplots()
            tour_coords = coords[best_route + [best_route[0]]]
            ax.plot(tour_coords[:, 0], tour_coords[:, 1], 'o-', label=f"Step {i}, Cost: {best_cost:.2f}")
            ax.set_title("TSP Hill Climbing")
            ax.legend()
            ax.axis("off")
            frame_path = f"frames/frame_{i:03d}.png"
            plt.savefig(frame_path)
            images.append(imageio.imread(frame_path))
            plt.close()

        costs.append(current_cost)

    imageio.mimsave("hill_climbing_tsp.gif", images, duration=0.5)
    return best_route, best_cost, costs

# Initial random route
initial_route = list(range(num_nodes))
np.random.shuffle(initial_route)

best_route, best_cost, costs = hill_climbing(initial_route)
print("Best route:", best_route)
print("Cost:", best_cost)

# Plot stats
plt.figure(figsize=(10, 5))
plt.plot(range(len(costs)), costs, marker='o', linestyle='-', color='blue', label='Tour Cost')
plt.title("Hill Climbing - Tour Cost over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Tour Cost")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("hill_climbing_stats.png")
plt.show()
