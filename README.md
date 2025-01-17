# LQGames

## Description

`LQGames` is a Python package for simulating and solving Linear Quadratic (LQ) games, suitable for both finite and infinite horizon problems. This package provides tools for modeling dynamic systems with multiple players, each aiming to minimize their individual costs based on the system’s states and control inputs. It includes functions for continuous-time Algebraic Riccati Equations (CARE) solutions, state evolution simulations, and control input calculations, making it ideal for advanced applications in control theory and game theory.

## Key Features

- **Infinite Horizon LQ Games**: Supports infinite horizon games, allowing for stable, long-term system evolution simulations and optimizations.
- **Finite Horizon LQ Games**: Solves finite-time LQ games with detailed time-step simulation, making it ideal for time-constrained scenarios.
- **Simulations of State and Control Evolution**: Perform time-step simulations of both system state trajectories and optimal control inputs.

## Installation

After publishing to PyPI, the package can be installed via pip:

```bash
pip install LQGames
```

## Usage

```python
from LQGames.LQGames import LQGame_Infnity, LQGame_Finity

# Initialize system matrices and parameters
A = ...  # Autonomous movement matrix
B = [...]  # List of input matrices for each player
Q = [...]  # List of state weight matrices for each player
R = [...]  # List of input weight matrices for each player
x_0 = ...  # Initial state vector

# Set up an infinite horizon LQ game
lq_game = LQGame_Infnity(A=A, B=B, Q=Q, R=R, x_0=x_0)
lq_game.Simulation_x()  # Simulate state evolution

# Access simulation results
states = lq_game.Simulation_x
```

## Requirements

- **Python 3.6+**
- **NumPy**
- **SciPy**
- **warning**
- **copy**

## License

This package is licensed under the MIT License.
