
from softpy.evolutionary import FloatVectorCandidate
import numpy as np 
from scipy.stats import uniform




'''
    notes:

        inertia was initially set as a fixed float at 0.7,
        by reserching online i dicovered an adaptive inertia which i decided to 
        use 
        it is worth to note that without the adaptive inertia the optimization of
        the rosenbrock function was never possible, the algorithm was stuck in local 
        optima close to the solution 

        on the other hand even with adaptive inertia a opimal solution of a function 
        like the Rastrigin is still not possile over every run 
        the reason being that "The Rastrigin function is a highly multimodal benchmark, 
                               with a large number of regularly spaced local minima due 
                               to its sinusoidal components."
        
        in this specific situation a way to solve the problem would be to do fitness sharing to avoid 
        premature convergence, but it is important to note that fitness sharing does not help convegence in case of functions similar to 
        the rosenbrack, on the other hand, it can be harmful, 
        
        to solve this problem my idea would be to add an extra hyperparameter (attribute) in the 
        pso algorithm which if set to true uses a fitness shared function to compute the fitness and if not it doesn t

        since the attribute to have are specified i will not add this hyperparameter 

'''








class ParticleCandidate(FloatVectorCandidate):

    '''
        
    Inherits:
        FloatVectorCandidate: Provides base functionality for candidate position representation
        with boundary constraints and distribution sampling.
    '''
    
    def __init__(self,
                 size: int,
                 lower: np.ndarray,
                 upper: np.ndarray,
                 candidate: np.ndarray,
                 velocity: np.ndarray,
                 inertia: float,
                 wl: float,
                 wn: float,
                 wg: float
                 ):


        super().__init__(size=size,
                         candidate=candidate,
                         distribution=uniform,  
                         lower=lower, 
                         upper=upper)  

      

       
        self.velocity = velocity
        self.inertia = inertia
        self.wl = wl
        self.wn = wn
        self.wg = wg





    def generate(size: int, lower: np.ndarray, upper: np.ndarray):

        ''' method used to create an instance of the class, since the class is build as a factory'''
       

    
        inertia=0   # dummy value for inertia, it will be changed in the fit loop
              
        candidate = np.random.uniform(low=lower, high=upper, size=size)

        delta = np.abs(upper - lower)
        velocity = np.random.uniform(low=-delta, high=delta, size=size)
        wl = 0.3
        wn = 0.3
        wg = 0.4

        return ParticleCandidate(size, lower, upper, candidate, velocity, inertia, wl, wn, wg)
    


    def mutate(self):
        #uppdated the position of the candidate
        
        self.candidate+=self.velocity 
        
        

        
    
    # the recombination was changed since in the project specification is defined in the opposite way 
    # (moving away from the optimum)

    def recombine(self, local_best, neighborhood_best, best):
        
        rl = np.random.uniform(0, 1, size=self.size)
        rn = np.random.uniform(0, 1, size=self.size)
        rg = np.random.uniform(0, 1, size=self.size)
        

        self.velocity = (self.inertia * self.velocity
                        + rl * self.wl * (local_best - self.candidate)
                        + rn * self.wn * (neighborhood_best - self.candidate)
                        + rg * self.wg * (best - self.candidate)
                        )
    #updates the velocity 


        
    
    '''
    The clone() method creates a deep copy of the current particle.
    used to preserve the best poistion discovered so far without them be affected by subsequent updates 
    '''

    
    def clone(self):
        return ParticleCandidate(
            size=self.size,
            lower=self.lower,
            upper=self.upper,
            candidate=self.candidate.copy(),
            velocity=self.velocity.copy(),
            inertia=self.inertia,
            wl=self.wl,
            wn=self.wn,
            wg=self.wg
        )




