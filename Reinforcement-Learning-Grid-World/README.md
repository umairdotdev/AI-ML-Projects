# Reinforcement Learning: Grid World Simulation

This project demonstrates a grid world simulation for reinforcement learning, supporting two methods:
- **Q-Learning**
- **Value Iteration**

## Overview
The program simulates an agent navigating a grid world with:
- **Obstacles**: Block the agent's movement.
- **Pitfalls**: Penalize the agent for stepping into them.
- **Goal**: Rewards the agent for reaching it.

### How It Works
The grid world is a grid with dimensions (M x N), where each cell represents a state. The agent starts in a random position and navigates the grid to maximize its cumulative reward.

#### Actions and Their Representations
Each cell in the grid has an action associated with it:
- **0**: Move **Up**.
- **1**: Move **Right**.
- **2**: Move **Down**.
- **3**: Move **Left**.

#### Special Cells
- **Obstacles (X)**: The agent cannot move through these cells.
- **Pitfalls (P)**: The agent is penalized if it steps into these cells.
- **Goal (G)**: The agent is rewarded for reaching this cell.
- **Agent (A)**: The starting position of the agent.

### Input Files
- **Q-Learning Input**: Specifies parameters such as learning rate, grid dimensions, obstacles, pitfalls, and rewards.
- **Value Iteration Input**: Specifies grid dimensions, convergence threshold, obstacles, pitfalls, and rewards.

### Output
The program generates a file mapping each grid cell to its optimal action.

## Usage
Run the program using:
```bash
python3 grid_world_simulation.py <input_file> <output_file>
```

### Example
Run Q-Learning:
```bash
python3 grid_world_simulation.py q_input.txt q_output.txt
```

Run Value Iteration:
```bash
python3 grid_world_simulation.py v_input.txt v_output.txt
```

### Files in the Project
- `grid_world_simulation.py`: The main script for the simulation.
- `q_input.txt`: Sample input file for Q-Learning.
- `v_input.txt`: Sample input file for Value Iteration.
- `q_learning_output_uploaded.txt`: Output file for Q-Learning.
- `value_iteration_output_uploaded.txt`: Output file for Value Iteration.
- `README.md`: Project documentation.

## Requirements
- Python 3
- NumPy

## Author
This project was authored by Umair Aslam and adapted for public use.
