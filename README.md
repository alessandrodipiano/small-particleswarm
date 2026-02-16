# Particle Swarm Optimization (PSO) — From-Scratch Implementation

This project implements Particle Swarm Optimization (PSO), a population-based metaheuristic for continuous optimization, developed from first principles and evaluated on standard benchmark functions.

The implementation emphasizes algorithmic clarity, modular design, and empirical evaluation on both convex and multimodal objective landscapes.

---

## Overview

Particle Swarm Optimization is inspired by collective behavior observed in biological systems such as bird flocks or fish schools. A population of candidate solutions ("particles") explores the search space by combining:

- Individual experience (personal best)
- Social information from neighbors
- Global knowledge of the swarm

PSO is widely used for solving nonlinear, nonconvex, and high-dimensional optimization problems.

---

## Algorithmic Framework

Each particle is defined by:

- Position vector in a continuous search space
- Velocity vector governing movement
- Personal best position found so far

At each iteration, the velocity is updated according to three attraction components:

1. Personal best (cognitive component)
2. Neighborhood best (social component)
3. Global best (collective component)

The position is then updated by applying the new velocity.

### Velocity Update Rule

v(t+1) = ω v(t)
         + r₁ c₁ (p_best − x)
         + r₂ c₂ (n_best − x)
         + r₃ c₃ (g_best − x)

where:

- ω is the inertia coefficient
- c₁, c₂, c₃ are acceleration weights
- r₁, r₂, r₃ are uniform random variables in [0, 1]
- p_best, n_best, g_best denote personal, neighborhood, and global best positions

### Position Update

x(t+1) = x(t) + v(t+1)

---

## Adaptive Inertia Strategy

To balance exploration and exploitation, the inertia weight decreases linearly over time:

ω(t) = 0.9 − 0.5 · (t / T)

This schedule encourages:

- Broad exploration during early iterations
- Fine local search near convergence

Empirically, adaptive inertia was essential for optimizing difficult functions such as Rosenbrock.

---

## Implementation Details

The project is implemented in Python with a modular architecture:

- **Particle representation** encapsulates state and update rules
- **Optimizer** manages population dynamics and best-solution tracking
- **Benchmark module** evaluates performance on test functions

Particles maintain boundary constraints and randomized initialization of both position and velocity.

---

## Benchmark Functions

The optimizer was tested on widely used optimization benchmarks.

### Convex Functions (Single Global Optimum)

**Sphere Function**

Smooth quadratic bowl with minimum at the origin.  
Used to verify basic convergence behavior.

**Manhattan Function**

Absolute-value variant with a non-differentiable optimum at the origin.  
Tests robustness to non-smooth landscapes.

---

### Multimodal Functions (Multiple Local Optima)

**Rastrigin Function**

Highly multimodal with many regularly spaced local minima.  
Challenges the optimizer’s ability to avoid premature convergence.

**Rosenbrock Function**

Nonconvex function with a narrow curved valley leading to the global optimum.  
Requires coordinated movement across dimensions.

---

## Experimental Observations

Empirical results highlight characteristic PSO behavior:

- Reliable convergence on convex functions
- Sensitivity to population size and iteration count on complex landscapes
- Rosenbrock optimization requires larger swarms and longer runs
- Rastrigin frequently traps particles in local minima
- Adaptive inertia significantly improves performance

These observations are consistent with known properties of swarm-based metaheuristics.

---

## Project Structure

candidate.py Particle representation and update mechanics
optimizer.py Core PSO algorithm
test-functions.py Experimental evaluation on benchmark functions
---

## Dependencies

Python 3.8 or later

Required packages:

- numpy
- scipy
- softpy

Install with:

pip install numpy scipy softpy

---

## Running the Experiments

Execute the benchmark script:

python test-functions.py

The script evaluates the optimizer across multiple functions and search domains and reports the best solutions found.

---

## Limitations

This implementation focuses on clarity rather than advanced performance features. Limitations include:

- Fixed topology for neighborhood selection
- No diversity preservation mechanisms
- No constraint handling beyond simple bounds
- Susceptibility to local optima in highly multimodal problems

---

## Future Work

Possible extensions include:

- Alternative neighborhood topologies
- Fitness sharing or niching techniques
- Constriction-factor PSO variants
- Hybrid algorithms combining PSO with local search
- Parallel or GPU implementations
- Support for constrained optimization problems

---

## Author

Alessandro Di Piano  
Bachelor’s student in Artificial Intelligence
