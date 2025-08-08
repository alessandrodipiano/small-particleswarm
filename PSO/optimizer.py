from softpy.evolutionary.singlestate import MetaHeuristicsAlgorithm
import numpy as np
from PSO.candidate import ParticleCandidate



class ParticleSwarmOptimizer(MetaHeuristicsAlgorithm):
   
   
   
    def __init__(self, fitness_func, pop_size: int, n_neighbors: int, **kwargs):
        self.fitness_func = fitness_func
        self.pop_size = pop_size
        self.n_neighbors = n_neighbors
        self.kwargs = kwargs

        self.best = [None] * pop_size
        self.global_best = None
        self.fitness_best = np.full(pop_size, -np.inf)
        self.global_fitness_best = -np.inf

    

        '''
        Initializes the swarm before starting optimization.
        
        - Generates the initial population of particles with random positions and velocities
            within the specified bounds.
        - Sets up arrays/lists to track personal best solutions and their fitness values.
        - Finds and stores the best particle and its fitness among the initial population.
        
        '''
    
    def initialize(self):
        self.population = np.array([
            ParticleCandidate.generate(**self.kwargs) for _ in range(self.pop_size)
        ])
        self.best = [None] * self.pop_size
        self.fitness_best = np.full(self.pop_size, -np.inf)
        self.global_best = None
        self.global_fitness_best = -np.inf

        for i, particle in enumerate(self.population):
            fitness = self.fitness_func(particle)
            self.best[i] = particle.clone()
            self.fitness_best[i] = fitness

            if fitness > self.global_fitness_best:
                self.global_best = particle.clone()
                self.global_fitness_best = fitness

    

    '''
        Executes the main optimization loop for a specified number of iterations 

        Steps:
        1. Initialization (if not already done): Ensures that the population is initialized with candidate solutions.
        2.
        - Each particle's fitness is evaluated using a specific fitness function.
        - If a particle's new fitness exceeds its historical best, update its personal best.
        - Update the global best solution found so far if the current particle exceeds the global fitness.
        3.
        - For each particle, a random neighborhood other particles is sampled.
        - The best individual in the neighborhood is selected based on stored fitness values.
        - The particle then undergoes a recombination step using:
            - its own personal best,
            - the neighborhood best,
            - and the global best.
       
    '''

    def fit(self, n_iters=1):
        if not hasattr(self, 'population') or self.population is None:
            self.initialize()
        
        
        
        for t in range(n_iters):
            
            inertia = 0.9 - 0.5 * (t / n_iters)

            for i, particle in enumerate(self.population):
                particle.inertia = inertia
                fitness = self.fitness_func(particle)

                if fitness > self.fitness_best[i]:
                    self.fitness_best[i] = fitness
                    self.best[i] = particle.clone()

                    if fitness > self.global_fitness_best:
                        self.global_best = particle.clone()
                        self.global_fitness_best = fitness

            for i, particle in enumerate(self.population):
                neighbor_indices = np.random.choice(
                    [j for j in range(self.pop_size) if j != i],
                    size=self.n_neighbors,
                    replace=False
                )
                best_neighbor_idx = max(
                    neighbor_indices,
                    key=lambda j: self.fitness_best[j]
                )
                best_neighbor = self.best[best_neighbor_idx]

                particle.recombine(
                    local_best=self.best[i].candidate,
                    neighborhood_best=best_neighbor.candidate,
                    best=self.global_best.candidate
                )
                particle.mutate()
