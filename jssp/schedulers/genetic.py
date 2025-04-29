"""
Genetic Algorithm implementation for Job Shop Scheduling Problems.
"""

import random
import time
import numpy as np
from typing import List, Dict, Tuple, Any, Optional

from jssp.data import JSSPInstance, Job, Operation
from jssp.schedulers import Scheduler


class Chromosome:
    """
    Represents a chromosome in the genetic algorithm.
    
    A chromosome is a permutation of operations across all jobs,
    where each job's operations maintain their relative order.
    """
    
    def __init__(self, instance: JSSPInstance, gene_sequence: Optional[List[Tuple[int, int]]] = None):
        self.instance = instance
        
        if gene_sequence is None:
            self.gene_sequence = self._create_random_sequence()
        else:
            self.gene_sequence = gene_sequence
            
        self.fitness = float('inf')  # Lower is better
        self.schedule_result = None
        
    def _create_random_sequence(self) -> List[Tuple[int, int]]:
        """Create a random valid permutation of operations."""
        # Create a list of (job_id, operation_idx) pairs
        all_operations = []
        
        for j, job in enumerate(self.instance.jobs):
            for op_idx in range(len(job.operations)):
                all_operations.append((j, op_idx))
        
        # Shuffle the list while respecting operation precedence
        sequence = []
        # Track the next operation index for each job
        next_op_idx = [0] * len(self.instance.jobs)
        
        while len(sequence) < len(all_operations):
            # Collect indices of jobs that have operations left
            eligible_jobs = [j for j in range(len(self.instance.jobs))
                           if next_op_idx[j] < len(self.instance.jobs[j].operations)]
            
            if not eligible_jobs:
                break
                
            # Choose a random job
            job_idx = random.choice(eligible_jobs)
            
            # Add the next operation of this job
            sequence.append((job_idx, next_op_idx[job_idx]))
            
            # Update the next operation index for this job
            next_op_idx[job_idx] += 1
        
        return sequence
    
    def decode_and_evaluate(self) -> Dict[str, Any]:
        """
        Decode the chromosome into a schedule and evaluate its fitness.
        
        Returns:
            A dictionary with scheduling metrics
        """
        # Reset the schedule
        self.instance.reset_schedule()
        
        # Keep track of when each job's last operation was completed
        job_completion_times = [0] * len(self.instance.jobs)
        
        # Process operations in the order specified by the chromosome
        for job_idx, op_idx in self.gene_sequence:
            job = self.instance.jobs[job_idx]
            operation = job.operations[op_idx]
            machine = self.instance.machines[operation.machine_id]
            
            # Calculate the earliest start time for this operation
            # It needs to be after:
            # 1. The job's previous operation is complete
            # 2. The machine becomes available
            earliest_start_time = max(
                job_completion_times[job_idx],
                machine.get_earliest_available_time()
            )
            
            # Schedule the operation
            machine.schedule_operation(job, operation, earliest_start_time)
            
            # Update the job's completion time
            job_completion_times[job_idx] = operation.end_time
        
        # Calculate fitness (makespan - lower is better)
        makespan = self.instance.makespan()
        self.fitness = makespan
        
        result = {
            "algorithm": "Genetic Algorithm",
            "makespan": makespan,
            "total_flow_time": self.instance.total_flow_time(),
            "average_flow_time": self.instance.average_flow_time(),
            "is_valid": self.instance.is_valid_schedule(),
        }
        
        self.schedule_result = result
        return result


class GeneticScheduler(Scheduler):
    """
    Genetic Algorithm scheduler for JSSP.
    
    Uses a permutation-based representation with crossover and mutation
    operators designed to maintain operation precedence constraints.
    """
    
    def __init__(self, 
                 population_size: int = 100, 
                 generations: int = 100,
                 crossover_prob: float = 0.8,
                 mutation_prob: float = 0.2,
                 elitism_count: int = 2,
                 tournament_size: int = 3):
        """
        Initialize the genetic algorithm parameters.
        
        Args:
            population_size: Size of the population
            generations: Number of generations to evolve
            crossover_prob: Probability of crossover
            mutation_prob: Probability of mutation
            elitism_count: Number of best individuals to keep unchanged
            tournament_size: Size of tournament for selection
        """
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.elitism_count = elitism_count
        self.tournament_size = tournament_size
    
    def schedule(self, instance: JSSPInstance) -> Dict[str, Any]:
        start_time = time.time()
        
        # Create initial population
        population = self._initialize_population(instance)
        
        # Evaluate the initial population
        for chromosome in population:
            chromosome.decode_and_evaluate()
        
        # Sort by fitness (makespan - lower is better)
        population.sort(key=lambda x: x.fitness)
        
        best_fitness_history = [population[0].fitness]
        avg_fitness_history = [sum(c.fitness for c in population) / len(population)]
        
        # Main evolutionary loop
        for generation in range(self.generations):
            # Create new population with elitism
            new_population = population[:self.elitism_count]
            
            # Fill the rest of the population
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)
                
                # Crossover
                if random.random() < self.crossover_prob:
                    offspring = self._crossover(parent1, parent2, instance)
                else:
                    # No crossover, clone one parent
                    offspring = Chromosome(instance, list(parent1.gene_sequence))
                
                # Mutation
                if random.random() < self.mutation_prob:
                    self._mutate(offspring)
                
                # Add to new population
                new_population.append(offspring)
            
            # Replace old population
            population = new_population
            
            # Evaluate the new population
            for chromosome in population:
                if chromosome.fitness == float('inf'):  # Only evaluate if not already evaluated
                    chromosome.decode_and_evaluate()
            
            # Sort by fitness
            population.sort(key=lambda x: x.fitness)
            
            # Track best and average fitness
            best_fitness_history.append(population[0].fitness)
            avg_fitness_history.append(sum(c.fitness for c in population) / len(population))
        
        # Return best solution found
        best_chromosome = population[0]
        best_chromosome.decode_and_evaluate()  # Ensure it's evaluated
        
        end_time = time.time()
        
        result = best_chromosome.schedule_result.copy()
        result.update({
            "algorithm": "Genetic Algorithm",
            "computation_time": end_time - start_time,
            "generations": self.generations,
            "population_size": self.population_size,
            "best_fitness_history": best_fitness_history,
            "avg_fitness_history": avg_fitness_history,
        })
        
        return result
    
    def _initialize_population(self, instance: JSSPInstance) -> List[Chromosome]:
        """Initialize a random population."""
        return [Chromosome(instance) for _ in range(self.population_size)]
    
    def _tournament_selection(self, population: List[Chromosome]) -> Chromosome:
        """Tournament selection."""
        tournament = random.sample(population, self.tournament_size)
        return min(tournament, key=lambda x: x.fitness)
    
    def _crossover(self, parent1: Chromosome, parent2: Chromosome, instance: JSSPInstance) -> Chromosome:
        """
        Precedence preserving crossover.
        
        Creates a child chromosome that preserves the relative order of operations
        for each job, while combining genetic material from both parents.
        """
        # Get the number of jobs and the sequence length
        n_jobs = len(instance.jobs)
        seq_length = len(parent1.gene_sequence)
        
        # Track which operations have been scheduled for each job
        scheduled_ops_count = [0] * n_jobs
        
        # Child's gene sequence
        child_sequence = []
        
        # Choose crossover points
        crossover_point1 = random.randint(0, seq_length - 1)
        crossover_point2 = random.randint(crossover_point1, seq_length - 1)
        
        # Copy the segment between crossover points from parent1
        middle_segment = parent1.gene_sequence[crossover_point1:crossover_point2+1]
        
        # Update the scheduled operations count
        for job_idx, op_idx in middle_segment:
            scheduled_ops_count[job_idx] += 1
        
        # Copy from parent2, maintaining precedence
        parent2_idx = 0
        
        # Fill the first part (before crossover_point1)
        while len(child_sequence) < crossover_point1:
            job_idx, op_idx = parent2.gene_sequence[parent2_idx]
            
            # Check if this operation can be added
            if scheduled_ops_count[job_idx] == op_idx:
                child_sequence.append((job_idx, op_idx))
                scheduled_ops_count[job_idx] += 1
            
            parent2_idx += 1
            if parent2_idx >= seq_length:
                break
        
        # Add the middle segment from parent1
        child_sequence.extend(middle_segment)
        
        # Continue with parent2 for the remainder
        while len(child_sequence) < seq_length and parent2_idx < seq_length:
            job_idx, op_idx = parent2.gene_sequence[parent2_idx]
            
            # Check if this operation can be added
            if scheduled_ops_count[job_idx] == op_idx:
                child_sequence.append((job_idx, op_idx))
                scheduled_ops_count[job_idx] += 1
            
            parent2_idx += 1
        
        # If there are still operations to be scheduled, add them in job order
        while len(child_sequence) < seq_length:
            for job_idx in range(n_jobs):
                if scheduled_ops_count[job_idx] < len(instance.jobs[job_idx].operations):
                    op_idx = scheduled_ops_count[job_idx]
                    child_sequence.append((job_idx, op_idx))
                    scheduled_ops_count[job_idx] += 1
                    break
        
        return Chromosome(instance, child_sequence)
    
    def _mutate(self, chromosome: Chromosome) -> None:
        """
        Swap mutation that preserves precedence constraints.
        
        Randomly selects two positions and swaps them if the operations
        can be exchanged without violating precedence constraints.
        """
        if len(chromosome.gene_sequence) <= 1:
            return
        
        # Try a limited number of times to find a valid swap
        for _ in range(10):  # Try at most 10 times
            # Choose two random positions
            pos1 = random.randint(0, len(chromosome.gene_sequence) - 1)
            pos2 = random.randint(0, len(chromosome.gene_sequence) - 1)
            
            if pos1 == pos2:
                continue
                
            # Make sure pos1 < pos2
            if pos1 > pos2:
                pos1, pos2 = pos2, pos1
            
            # Get the operations at these positions
            job1, op1 = chromosome.gene_sequence[pos1]
            job2, op2 = chromosome.gene_sequence[pos2]
            
            # Check if they're from different jobs
            if job1 == job2:
                continue  # Can't swap operations from the same job (precedence violation)
            
            # Check the operations immediately before and after the swap positions
            valid_swap = True
            
            # Check operations between pos1 and pos2
            for i in range(pos1 + 1, pos2):
                job_i, op_i = chromosome.gene_sequence[i]
                
                # If there's an operation from job1 with index <= op1, swap is invalid
                if job_i == job1 and op_i < op1:
                    valid_swap = False
                    break
                
                # If there's an operation from job2 with index >= op2, swap is invalid
                if job_i == job2 and op_i > op2:
                    valid_swap = False
                    break
            
            if valid_swap:
                # Perform the swap
                chromosome.gene_sequence[pos1], chromosome.gene_sequence[pos2] = \
                    chromosome.gene_sequence[pos2], chromosome.gene_sequence[pos1]
                
                # Mark chromosome for re-evaluation
                chromosome.fitness = float('inf')
                return 