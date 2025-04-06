import imageio
import gym
from travelling_salesman import hill_climb
# Create environment with rendering
env = gym.make("TSP-v0", render_mode="rgb_array", number_of_nodes=10)

# Run and record
best_tour, best_cost, costs, frames = hill_climb(env, record=True)

# Save GIF
imageio.mimsave("tsp_hill_climbing.gif", frames, duration=0.5)

print("GIF saved as tsp_hill_climbing.gif")
