import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
from TSP_env import TSPEnv

# Parameters
num_nodes = 20
env = TSPEnv(num_nodes=num_nodes, batch_size=1, num_draw=1)
state = env.reset()[0]
coords = state[:, :2]

# Distance calculation
def total_distance(route):
    dist = 0
    for i in range(len(route)):
        dist += np.linalg.norm(coords[route[i]] - coords[route[(i + 1) % len(route)]])
    return dist

# Simulated Annealing
def simulated_annealing(init_route, iterations=1000, initial_temp=100.0, cooling_rate=0.995):
    current_route = init_route.copy()
    current_cost = total_distance(current_route)
    best_route = current_route.copy()
    best_cost = current_cost
    temp = initial_temp
    images = []
    costs = []

    os.makedirs("frames_sa", exist_ok=True)

    for i in range(iterations):
        a, b = np.random.choice(len(current_route), size=2, replace=False)
        new_route = current_route.copy()
        new_route[a], new_route[b] = new_route[b], new_route[a]
        new_cost = total_distance(new_route)

        # Acceptance condition
        if new_cost < current_cost or np.random.rand() < np.exp((current_cost - new_cost) / temp):
            current_route = new_route
            current_cost = new_cost

            if new_cost < best_cost:
                best_route = new_route
                best_cost = new_cost

        temp *= cooling_rate
        costs.append(current_cost)

        if i % 50 == 0 or i == iterations - 1:
            fig, ax = plt.subplots()
            tour_coords = coords[best_route + [best_route[0]]]
            ax.plot(tour_coords[:, 0], tour_coords[:, 1], 'o-', label=f"Step {i}, Cost: {best_cost:.2f}")
            ax.set_title("TSP Simulated Annealing")
            ax.legend()
            ax.axis("off")
            frame_path = f"frames_sa/frame_{i:03d}.png"
            plt.savefig(frame_path)
            images.append(imageio.imread(frame_path))
            plt.close()

    imageio.mimsave("simulated_annealing_tsp.gif", images, duration=0.5)
    return best_route, best_cost, costs

# Initial random route
initial_route = list(range(num_nodes))
np.random.shuffle(initial_route)

# Run Simulated Annealing
best_route, best_cost, costs = simulated_annealing(initial_route)
print("Best route (Simulated Annealing):", best_route)
print("Cost:", best_cost)

# Plot stats
plt.figure(figsize=(10, 5))
plt.plot(range(len(costs)), costs, marker='o', linestyle='-', color='green', label='Tour Cost (SA)')
plt.title("Simulated Annealing - Tour Cost over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Tour Cost")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("simulated_annealing_stats.png")
plt.show()
