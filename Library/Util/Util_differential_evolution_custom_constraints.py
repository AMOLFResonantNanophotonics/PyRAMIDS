"""
#Custom Differential Evolution optimizer with geometric constraints

#This file provides a Differential Evolution implementation with:
#(1) minimum-separation constraints for scatterers,
#(2) optional exclusion-zone constraints,
#(3) optional caching of merit evaluations,
#(4) generation-by-generation save to output npz.

#Used in optimization workflows (e.g. inverse-design examples).
#
#Main user class
#  - DifferentialEvolution
#    initialize with bounds/objective and call `optimize()`

@author: dpal,fkoenderink
"""

import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
import os

class DifferentialEvolution:
    """Differential Evolution optimizer with geometric constraints.

    Intuition
    ---------
    Designed for scatterer-position optimization where each individual is
    `[x1, y1, x2, y2, ...]`, with optional minimum-separation and exclusion
    zone constraints.
    """
    def __init__(self,
        bounds,
        pop_size,
        max_generations,
        merit_func,
        args,
        outputFilename,
        dipole_locations,
        num_cores,
        F=0.5,
        CR=0.7,
        min_separation=None,
        initial_population=None,
        step_size=0.001,  # Default step_size to 0.001
        cache=True,       # Default cache to True
        exclude_square_side=None,   # e.g. 0.25  (microns)
        exclude_margin=0.0,         # optional padding
    ):
        """Create optimizer and initialize population.

        Parameters
        ----------
        bounds : sequence
            Per-dimension `(min, max)` bounds.
        pop_size : int
            Population size.
        max_generations : int
            Maximum DE generations.
        merit_func : callable
            Objective function `f(x, *args)` to minimize.
        outputFilename : str
            `.npz` file path used for progress/history saving.
        min_separation : float, optional
            Minimum allowed inter-particle distance.
        initial_population : array-like, optional
            Seed individuals.
        """
        
        self.bounds = np.array(bounds)
        self.pop_size = pop_size
        self.dimensions = len(bounds)
        self.max_generations = max_generations
        self.merit_func = merit_func
        self.args = args
        self.outputFilename = outputFilename
        self.dipole_locations = dipole_locations
        self.num_cores = num_cores
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability
        self.min_separation = min_separation  # Minimum separation constraint
        self.initial_population = initial_population  # Optional initial population
        self.step_size = step_size  # Rounding step size
        self.cache = cache  # Whether to use caching. helps speeding up
        self.exclude_square_side = exclude_square_side
        self.exclude_margin = exclude_margin

        if self.cache:
            self.fom_cache = {}  # Initialize cache if caching is enabled
        else:
            self.fom_cache = None  # No cache

        # Initialize optimization history
        self.fom_hist = []  # To store figure of merit history
        self.pop_hist = []  # To store population history
        self.best_score = np.inf
        self.best_position = None

        # Initialize population
        if self.initial_population is not None:
            # Use provided initial positions and ensure the separation constraint is respected
            self.population = self.initialize_positions_with_constraints(self.initial_population)
        else:
            # Generate random positions within bounds that satisfy the constraints
            self.population = self.generate_positions_with_constraints()

    def round_to_step(self, value):
        """Round values to configured discretization step size."""
        return np.round(value / self.step_size) * self.step_size


    def initialize_positions_with_constraints(self, initial_positions):
        """Initialize positions with provided positions and enforce constraints."""
        initial_positions = np.array(initial_positions).reshape(-1, self.dimensions)

        # If fewer initial positions are provided, replicate and perturb
        if initial_positions.shape[0] < self.pop_size:
            # Replicate initial positions to reach population size
            num_repeats = int(np.ceil(self.pop_size / initial_positions.shape[0]))
            initial_positions = np.tile(initial_positions, (num_repeats, 1))
            # Trim to pop_size
            initial_positions = initial_positions[:self.pop_size]

            # Calculate perturbation radius
            radius = self.min_separation / 2

            num_particles = self.dimensions // 2  # Number of particles per individual

            # Generate random angles between 0 and 2π
            angles = np.random.uniform(0, 2 * np.pi, (self.pop_size, num_particles))

            # Generate random distances (uniformly within circle)
            u = np.random.uniform(0, 1, (self.pop_size, num_particles))
            distances = radius * np.sqrt(u)

            # Compute x and y perturbations
            delta_x = distances * np.cos(angles)
            delta_y = distances * np.sin(angles)

            # Combine perturbations
            perturbations = np.empty((self.pop_size, self.dimensions))
            perturbations[:, ::2] = delta_x
            perturbations[:, 1::2] = delta_y

            # Apply perturbations
            perturbed_positions = initial_positions + perturbations

        elif initial_positions.shape[0] == self.pop_size:
            # Use initial positions as is (without perturbations)
            perturbed_positions = initial_positions.copy()
        else:
            # If more initial positions are provided than pop_size, use the pop_size
            perturbed_positions = initial_positions[:self.pop_size].copy()

        # Apply separation constraints to each individual
        for i in range(self.pop_size):
            perturbed_positions[i] = self.apply_separation_constraint_to_individual(
                perturbed_positions[i])

        # Enforce xy 2d bounds
        perturbed_positions = np.clip(perturbed_positions, self.bounds[:, 0], self.bounds[:, 1])

        # # Round positions using the instance method (if applicable)
        # perturbed_positions = self.round_to_step(perturbed_positions)
        return perturbed_positions



    def generate_positions_with_constraints(self):
        """Generate random positions within bounds that satisfy the separation constraint."""
        positions = np.zeros((self.pop_size, self.dimensions))
        num_particles = self.dimensions // 2  # Assuming each particle has x and y coordinates

        for i in range(self.pop_size):
            valid = False
            attempts = 0
            while not valid and attempts < 1000:
                # Generate random positions for particles within bounds
                individual = np.zeros(self.dimensions)
                individual[::2] = np.random.uniform(
                    self.bounds[::2, 0], self.bounds[::2, 1], num_particles)

                individual[1::2] = np.random.uniform(
                    self.bounds[1::2, 0], self.bounds[1::2, 1], num_particles)

                # Apply separation constraint to individual
                individual = self.apply_separation_constraint_to_individual(individual)

                # Enforce bounds
                individual = np.clip(individual, self.bounds[:, 0], self.bounds[:, 1])

                # # Round positions using the instance method
                # individual = self.round_to_step(individual)

                # Check if individual is valid (no overlaps)
                valid = True  # Since the constraint function ensures validity
                attempts += 1

            if not valid:
                raise ValueError("Could not generate valid initial population.")
            positions[i] = individual

        return positions



    def apply_separation_constraint_to_individual(self, individual):
        """Ensure particles within an individual maintain minimum separation."""
        
        # Copying the individual, original is unaltered
        adjusted_individual = individual.copy()

        num_particles = self.dimensions // 2  # Assuming each particle has x and y coordinates

        # Reshape positions for easier manipulation
        xy_positions = adjusted_individual.reshape(num_particles, 2)

        for _ in range(100):  # Limit iterations to prevent infinite loops
            # Compute pairwise distances
            dist_matrix = squareform(pdist(xy_positions))
            np.fill_diagonal(dist_matrix, np.inf)  # Ignore self-distances

            # Find overlapping pairs
            overlaps = dist_matrix < self.min_separation
            if not np.any(overlaps):
                break  # No overlaps, constraint is satisfied

            # Get indices of overlapping pairs
            i_indices, j_indices = np.where(overlaps)

            # Avoid duplicate pairs
            mask = i_indices < j_indices
            i_indices = i_indices[mask]
            j_indices = j_indices[mask]

            if len(i_indices) == 0:
                break

            # Computing movement to separate overlapping particles
            # Positions of overlapping particles
            p_i = xy_positions[i_indices]
            p_j = xy_positions[j_indices]

            # Compute direction vectors and distances
            directions = p_i - p_j
            distances = np.linalg.norm(directions, axis=1, keepdims=True) + 1e-9
            unit_directions = directions / distances

            # Compute movement required
            move_distances = ((self.min_separation - distances) / 2)
            move_vectors = move_distances * unit_directions

            # Initialize movement accumulator
            move_totals = np.zeros_like(xy_positions)

            # Apply and accumulate movements
            np.add.at(move_totals, i_indices, move_vectors)
            np.subtract.at(move_totals, j_indices, move_vectors)

            # Update positions
            xy_positions += move_totals

            # Enforce bounds with np.clip
            xy_positions[:, 0] = np.clip(
                xy_positions[:, 0], self.bounds[::2, 0], self.bounds[::2, 1]
            )
            xy_positions[:, 1] = np.clip(
                xy_positions[:, 1], self.bounds[1::2, 0], self.bounds[1::2, 1]
            )
            
            xy_positions = self.push_out_of_exclusion_square(xy_positions)

        # Flatten positions back to original shape
        adjusted_individual = xy_positions.flatten()

        # Round positions using the instance method
        adjusted_individual = self.round_to_step(adjusted_individual)
        
        # xy_positions = adjusted_individual.reshape(num_particles, 2)
        # # one last quick repair pass if rounding re-introduced overlaps
        # if np.any(pdist(xy_positions) < self.min_separation):
        #     # run a few more iterations of the same loop by recursion or a short repeat
        #     return self.apply_separation_constraint_to_individual(adjusted_individual)

        return adjusted_individual

    def push_out_of_exclusion_square(self, xy_positions):
    
        """
        Enforce: no particle is allowed inside a central square of side 'exclude_square_side'
        centered at origin. If a point falls inside, push it to the nearest edge + margin.
        """
        if self.exclude_square_side is None:
            return xy_positions
    
        half = 0.5 * self.exclude_square_side + self.exclude_margin
    
        x = xy_positions[:, 0]
        y = xy_positions[:, 1]
    
        inside = (np.abs(x) <= half) & (np.abs(y) <= half)
        if not np.any(inside):
            return xy_positions
    
        # For each inside point, push it to the nearest boundary (in L sense)
        xi = x[inside]
        yi = y[inside]
    
        # Which boundary is closer: vertical (|x|) or horizontal (|y|)
        push_x = (half - np.abs(xi)) <= (half - np.abs(yi))
    
        # Push to x = ±half or y = ±half, preserving sign (choose + if exactly 0)
        sign_x = np.where(xi >= 0, 1.0, -1.0)
        sign_y = np.where(yi >= 0, 1.0, -1.0)
    
        xi_new = xi.copy()
        yi_new = yi.copy()
    
        xi_new[push_x] = sign_x[push_x] * half
        yi_new[~push_x] = sign_y[~push_x] * half
    
        xy_positions[inside, 0] = xi_new
        xy_positions[inside, 1] = yi_new

        return xy_positions

    def optimize(self):
        """Run Differential Evolution iterations and persist history to disk."""
        for generation in tqdm(range(self.max_generations), desc="DE Progress"):
            
            # Evaluate FOM for all individuals
            scores = self.evaluate_population(self.population)
            print(f"Completed FOM evaluations for generation {generation+1}")

            # Update best solution
            min_idx = np.argmin(scores)
            if scores[min_idx] < self.best_score:
                self.best_score = scores[min_idx]
                self.best_position = self.population[min_idx].copy()

            # Vectorized mutation and crossover
            all_indices = np.arange(self.pop_size)

            # Choose 3 indices for mutation
            idx_choices = np.array(
                [np.random.choice(np.delete(all_indices, i), 3, replace=False) for i in all_indices])

            a_idx = idx_choices[:, 0]
            b_idx = idx_choices[:, 1]
            c_idx = idx_choices[:, 2]

            a = self.population[a_idx]
            b = self.population[b_idx]
            c = self.population[c_idx]

            mutants = a + self.F * (b - c)

            # Enforce bounds on mutants using np.clip
            mutants = np.clip(mutants, self.bounds[:, 0], self.bounds[:, 1])

            # Round mutants using the instance method
            mutants = self.round_to_step(mutants)

            # Crossover
            cross_probs = np.random.rand(self.pop_size, self.dimensions)
            rand_indices = np.random.randint(0, self.dimensions, self.pop_size)
            cross_masks = cross_probs < self.CR
            cross_masks[np.arange(self.pop_size), rand_indices] = True  # Ensure at least one parameter is from mutant

            trials = np.where(cross_masks, mutants, self.population)

            # Apply separation constraints to trials
            trials = np.array([
                self.apply_separation_constraint_to_individual(trial) for trial in trials])

            # Enforce bounds on trials
            trials = np.clip(trials, self.bounds[:, 0], self.bounds[:, 1])

            # Round trials using the instance method
            trials = self.round_to_step(trials)

            # Evaluate new population
            new_scores = self.evaluate_population(trials)

            # Selection
            improved = new_scores < scores
            self.population[improved] = trials[improved]
            scores[improved] = new_scores[improved]

            # Update best solution if improved
            min_idx = np.argmin(scores)
            if scores[min_idx] < self.best_score:
                self.best_score = scores[min_idx]
                self.best_position = self.population[min_idx].copy()

            # Record optimization history
            self.fom_hist.append(scores.copy())
            self.pop_hist.append(self.population.copy())

            # Save the output every few (here 1) generations
            if generation % 1 == 0:

                try:
                    output_dir = os.path.dirname(self.outputFilename)
                    if output_dir:
                        os.makedirs(output_dir, exist_ok=True)
                    temp_filename = self.outputFilename + ".tmp.npz"
                    
                    np.savez_compressed(
                        temp_filename,
                        dipole_locations=self.dipole_locations,
                        initial_pop=self.initial_population,
                        fom_hist=np.array(self.fom_hist),
                        pop_hist=np.array(self.pop_hist),
                    )
                    os.replace(temp_filename, self.outputFilename)  # Rename temp file to final filename
                except Exception as e:
                    print(f"Error saving file: {e}")
                    try:
                        if os.path.exists(temp_filename):
                            os.remove(temp_filename)
                    except Exception:
                        pass


            print(f"Generation {generation+1}/{self.max_generations}, Best Score: {self.best_score}")

            #Early stopping criterion
            if generation > 3000:
                recent_foms = np.array(self.fom_hist[-5:])
                if np.std(recent_foms) < 1e-5:
                    print("Convergence criteria met. Stopping optimization.")
                    break

        print(f"DE Optimization complete. Best Score: {self.best_score}")
        print("Best positions saved in:", self.outputFilename)



    def evaluate_population(self, population):
        """Evaluate objective for all individuals (with optional caching)."""
        scores = []
        for indiv in population:
            # Round positions using the instance method
            indiv = self.round_to_step(indiv)

            # If caching is enabled
            if self.cache:
                # Convert positions to a hashable key
                key = tuple(indiv)
                if key in self.fom_cache:
                    # Retrieve FOM from cache
                    score = self.fom_cache[key]
                else:
                    # Compute the FOM
                    score = self.merit_func_wrapper(indiv)
                    # Store the FOM in the cache
                    self.fom_cache[key] = score
            else:
                # Compute the FOM without caching
                score = self.merit_func_wrapper(indiv)

            scores.append(score)
        return np.array(scores)



    def merit_func_wrapper(self, x):
        """Internal objective wrapper."""
        # x is already rounded to the desired precision
        return self.merit_func(x, *self.args)
