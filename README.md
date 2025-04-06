# AI_Algorithms

This repository implements four classical AI search/optimization algorithms and evaluates their performance on specific environments. The project is part of an academic assignment requiring performance analysis, visual demonstration, and heuristic explanation.

---

## Algorithms Implemented

1. **Branch and Bound (BnB)**
2. **Iterative Deepening A\* (IDA\*)**
3. **Hill Climbing (HC)**
4. **Simulated Annealing (SA)**

---

## Environments Used

| Algorithm          | Environment      |
|--------------------|------------------|
| Branch and Bound   | Frozen Lake, Ant Maze |
| IDA\*              | Frozen Lake, Ant Maze |
| Hill Climbing      | Traveling Salesman Problem (TSP) |
| Simulated Annealing| Traveling Salesman Problem (TSP) |

---

## Performance Evaluation

Each algorithm was tested **five times** on its respective environment. The following performance metrics were recorded:

- **Time Taken** to reach the goal state or solution
- **Reward Obtained** (for Frozen Lake / Ant Maze)
- **Point of Convergence** (if applicable)
- **Termination Timeout (τ)**: Set to **10 minutes**. Runs are forcefully terminated after this time limit.

### Reproducibility

- Random seeds used where applicable
- Configurable timeout and run count for testing
- Results are averaged and plotted

---

## Demonstration

Visual GIFs of the algorithms running in their respective environments are included in the `media/` directory and in the accompanying slide deck.

---

## Heuristic Functions Used

- **Branch and Bound / IDA\***: Manhattan distance for grid-based environments like Frozen Lake and Ant Maze.
- **Hill Climbing / Simulated Annealing**: Inverse of total distance for TSP (objective: minimize tour length).

---

## Results

Plots for:
- Average time taken
- Success/failure rates within the τ-limit
- Convergence curves (if applicable)

Refer to the `results/` folder for raw data and generated plots.

---

## Setup
git clone https://github.com/i-apex/AI_Algorithms.git

cd AI_Algorithms

pip install -r requirements.txt

text

## Usage
### Run All Tests (5 iterations)

---

## Team
- [Jaimin Viramgama](https://github.com/i-apex)  
- [Siddhant Chatse](https://github.com/sid1309)
