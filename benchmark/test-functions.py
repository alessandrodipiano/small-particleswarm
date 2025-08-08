import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PSO.optimizer import ParticleSwarmOptimizer





# Fitness functions (all negated for maximization): 
# - Manhattan: convex, symmetric, minimum at origin. 
# - Sphere: convex, smooth, minimum at origin. 
# - Rastrigin: highly multimodal, non-convex, global minimum at origin. 
# - Rosenbrock: non-convex, curved valley, minimum at [1, ..., 1].





'''
we can see how the algorithm works perfectly in case of functions
who have only one global optimum

the 2 function use for this test are the sphere and manhattan, and the upper and lower bounds choosen are 
a standard one which contains the solution and one which does not, this to see if the algorithm could escape 
the initial initialization and explore the function globally 

the standard domain was found here https://www.sfu.ca/~ssurjano/spheref.html
'''




size=5 # the size of the candidate we will use 

def sphere(candidate):
    return -np.sum(candidate.candidate ** 2)

def manhattan_fitness(candidate):
    x = candidate.candidate
    return -np.sum(np.abs(x))

fitness_functions = [
    ("Sphere Function", sphere),
    ("Manhattan Fitness", manhattan_fitness),
]

search_spaces = [
    ("Negative Domain", np.full(size, -200), np.full(size, -100)),
    ("staandard domain", np.full(size, -5.12), np.full(size, 5.12))
]

for name, func in fitness_functions:
    for domain_name, lower, upper in search_spaces:
        print(f"=== {name} on {domain_name} ===")
        
        optimizer = ParticleSwarmOptimizer(
            fitness_func=func,
            pop_size=30,
            n_neighbors=5,
            size=size,
            lower=lower,
            upper=upper
        )

        optimizer.fit(n_iters=100)

        print(f"Best Fitness: {optimizer.global_fitness_best}")
        print(f"Best Candidate: {optimizer.global_best.candidate}")
        print()





'''while in case of multidimensional function with different local 
    optimum the results are a bit different, first of all we need more iterations and a bigger 
     population, indeed by using the same parameter as for the above 2 functions we can t get of course optimum
      results
    
    with enough iteration and a big population as stated at the start rosenbrack function converges for both the domain 

    while the rastrigin function does still not reach the optimum in every run and get stuck in a local valley most of the time 
      '''





size=5 # the size of the candidate we will use 

def rastrigin_fitness(candidate):
    x = candidate.candidate
    d = len(x)
    A = 10
    return - (A * d + np.sum(x**2 - A * np.cos(2 * np.pi * x)))

def rosenbrock_fitness(candidate):
    x = candidate.candidate
    return -np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

fitness_functions = [
    ("Rastrigin Function", rastrigin_fitness),
    ("Rosenbrock Function", rosenbrock_fitness),
]

search_spaces = [
    ("Negative Domain", np.full(size, -200), np.full(size, -100)),
    ("Standard Domain", np.full(5, -5), np.full(5, 5))
]

for name, func in fitness_functions:
    for domain_name, lower, upper in search_spaces:
        print(f"=== {name} on {domain_name} ===")
        
        optimizer = ParticleSwarmOptimizer(
            fitness_func=func,
            pop_size=130,
            n_neighbors=5,
            size=size,
            lower=lower,
            upper=upper
        )

        optimizer.fit(n_iters=1000)

        print(f"Best Fitness: {optimizer.global_fitness_best}")
        print(f"Best Candidate: {optimizer.global_best.candidate}")
        print() 