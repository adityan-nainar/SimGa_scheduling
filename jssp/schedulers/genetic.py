"""
Genetic Algorithm implementation for Job Shop Scheduling Problems.
"""

import random
import time
import numpy as np
import heapq
from typing import List, Dict, Tuple, Any, Optional

from jssp.data import JSSPInstance, Job, Operation
from jssp.schedulers import Scheduler, Event, EventType

# Special token for buffer insertion
BUFFER_TOKEN = (-1, -1)  # Represents a buffer/delay in the schedule

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
        for gene in self.gene_sequence:
            # Check if this is a buffer token
            if gene == BUFFER_TOKEN:
                # Add a small delay (buffer) - can be parameterized
                delay = random.randint(1, 5)
                # No action needed here, just skip and continue
                continue
                
            job_idx, op_idx = gene
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
                 buffer_mutation_prob: float = 0.1,  # Probability of buffer insertion mutation
                 elitism_count: int = 2,
                 tournament_size: int = 3,
                 use_uncertainty: bool = False,
                 num_simulations: int = 5,  # Number of simulations per fitness evaluation
                 stability_weight: float = 0.5):  # Weight for standard deviation in fitness
        """
        Initialize the genetic algorithm parameters.
        
        Args:
            population_size: Size of the population
            generations: Number of generations to evolve
            crossover_prob: Probability of crossover
            mutation_prob: Probability of mutation
            buffer_mutation_prob: Probability of buffer insertion mutation
            elitism_count: Number of best individuals to keep unchanged
            tournament_size: Size of tournament for selection
            use_uncertainty: Whether to consider uncertainty in fitness evaluation
            num_simulations: Number of simulations per fitness evaluation (K)
            stability_weight: Weight of standard deviation in fitness (alpha)
        """
        super().__init__(use_uncertainty, num_simulations, stability_weight)
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.buffer_mutation_prob = buffer_mutation_prob
        self.elitism_count = elitism_count
        self.tournament_size = tournament_size
    
    def schedule(self, instance: JSSPInstance) -> Dict[str, Any]:
        start_time = time.time()
        
        # Create initial population
        population = self._initialize_population(instance)
        
        # Evaluate the initial population
        for chromosome in population:
            chromosome.decode_and_evaluate()
            
            # If using uncertainty, evaluate with multiple simulations
            if self.use_uncertainty:
                # We need to copy the instance since evaluate_with_uncertainty will modify it
                instance_copy = chromosome.instance.copy()
                uncertainty_results = self.evaluate_with_uncertainty(instance_copy)
                chromosome.fitness = uncertainty_results["robust_fitness"]
                chromosome.schedule_result.update(uncertainty_results)
        
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
                
                # Regular mutation
                if random.random() < self.mutation_prob:
                    self._mutate(offspring)
                
                # Buffer insertion mutation
                if random.random() < self.buffer_mutation_prob:
                    self._buffer_mutation(offspring)
                
                # Add to new population
                new_population.append(offspring)
            
            # Replace old population
            population = new_population
            
            # Evaluate the new population
            for chromosome in population:
                if chromosome.fitness == float('inf'):  # Only evaluate if not already evaluated
                    chromosome.decode_and_evaluate()
                    
                    # If using uncertainty, evaluate with multiple simulations
                    if self.use_uncertainty:
                        instance_copy = chromosome.instance.copy()
                        uncertainty_results = self.evaluate_with_uncertainty(instance_copy)
                        chromosome.fitness = uncertainty_results["robust_fitness"]
                        chromosome.schedule_result.update(uncertainty_results)
            
            # Sort by fitness
            population.sort(key=lambda x: x.fitness)
            
            # Record history
            best_fitness_history.append(population[0].fitness)
            avg_fitness_history.append(sum(c.fitness for c in population) / len(population))
        
        # Return the best solution found
        best_solution = population[0]
        best_solution.decode_and_evaluate()  # Make sure the schedule is built
        
        # If using uncertainty, do a final evaluation with more simulations
        if self.use_uncertainty:
            final_instance = instance.copy()
            for j_idx, op_idx in best_solution.gene_sequence:
                if j_idx == -1:  # Buffer token
                    continue
                job = final_instance.jobs[j_idx]
                operation = job.operations[op_idx]
                # Apply the final schedule
                # (This is a simplified approach - in practice, we'd use the event simulation)
            
            final_results = self.evaluate_with_uncertainty(final_instance)
            best_solution.schedule_result.update(final_results)
        
        result = best_solution.schedule_result.copy()
        result.update({
            "computation_time": time.time() - start_time,
            "best_fitness_history": best_fitness_history,
            "avg_fitness_history": avg_fitness_history,
            "generations": self.generations
        })
        
        return result
    
    def _initialize_population(self, instance: JSSPInstance) -> List[Chromosome]:
        """Initialize a population of random chromosomes."""
        return [Chromosome(instance) for _ in range(self.population_size)]
    
    def _tournament_selection(self, population: List[Chromosome]) -> Chromosome:
        """Select a chromosome using tournament selection."""
        tournament = random.sample(population, min(self.tournament_size, len(population)))
        return min(tournament, key=lambda x: x.fitness)
    
    def _crossover(self, parent1: Chromosome, parent2: Chromosome, instance: JSSPInstance) -> Chromosome:
        """
        Perform precedence-preserving crossover (PPX) between two parents.
        
        This crossover ensures that the precedence relationships between
        operations of the same job are preserved in the offspring.
        """
        # Extract the sequence without buffer tokens
        p1_seq = [gene for gene in parent1.gene_sequence if gene != BUFFER_TOKEN]
        p2_seq = [gene for gene in parent2.gene_sequence if gene != BUFFER_TOKEN]
        
        # Initialize child sequence
        child_sequence = []
        
        # Track available operations for each job
        available_op_idx = [0] * len(instance.jobs)
        
        # Create a list of which parent to choose from at each step
        parent_choice = [random.choice([0, 1]) for _ in range(len(p1_seq))]
        
        # Build the child sequence
        for i in range(len(p1_seq)):
            parent_seq = p1_seq if parent_choice[i] == 0 else p2_seq
            
            # Find the next available operation from this parent
            for j, (job_idx, op_idx) in enumerate(parent_seq):
                # Skip if we've already used this operation
                if op_idx < available_op_idx[job_idx]:
                    continue
                    
                # Skip if this operation doesn't match the next available for this job
                if op_idx != available_op_idx[job_idx]:
                    continue
                    
                # We found a valid next operation
                child_sequence.append((job_idx, op_idx))
                available_op_idx[job_idx] += 1
                
                # Remove this operation from the parent sequence
                parent_seq = parent_seq[:j] + parent_seq[j+1:]
                if parent_choice[i] == 0:
                    p1_seq = parent_seq
                else:
                    p2_seq = parent_seq
                    
                break
        
        # Create and return the child chromosome
        return Chromosome(instance, child_sequence)
    
    def _mutate(self, chromosome: Chromosome) -> None:
        """
        Perform a swap mutation that preserves operation precedence.
        
        Randomly selects two operations from different jobs and swaps them
        if the precedence constraints allow it.
        """
        if len(chromosome.gene_sequence) <= 1:
            return
            
        # Filter out buffer tokens
        valid_indices = [i for i, gene in enumerate(chromosome.gene_sequence) 
                        if gene != BUFFER_TOKEN]
        
        if len(valid_indices) <= 1:
            return
            
        # Try several times to find a valid swap
        for _ in range(10):  # Limit attempts to avoid infinite loop
            # Pick two random positions
            i = random.choice(valid_indices)
            j = random.choice(valid_indices)
            
            # Ensure i < j
            if i > j:
                i, j = j, i
                
            if i == j:
                continue
                
            # Get the operations
            op1 = chromosome.gene_sequence[i]
            op2 = chromosome.gene_sequence[j]
            
            # Skip if same job (to maintain precedence)
            if op1[0] == op2[0]:
                continue
                
            # Check if swap is valid (no precedence violations)
            job1_ops_between = [idx for idx, (job_idx, _) in enumerate(chromosome.gene_sequence) 
                              if i < idx < j and job_idx == op1[0]]
                              
            job2_ops_between = [idx for idx, (job_idx, _) in enumerate(chromosome.gene_sequence)
                              if i < idx < j and job_idx == op2[0]]
            
            # If there are operations of the same job between our swap points, skip
            if job1_ops_between or job2_ops_between:
                continue
                
            # Perform the swap
            chromosome.gene_sequence[i], chromosome.gene_sequence[j] = (
                chromosome.gene_sequence[j], chromosome.gene_sequence[i]
            )
            return  # Successfully mutated
    
    def _buffer_mutation(self, chromosome: Chromosome) -> None:
        """
        Insert a buffer (delay) at a random position in the chromosome.
        
        This introduces deliberate idle time which can be beneficial 
        when dealing with uncertainty.
        """
        # Choose a random position to insert the buffer
        pos = random.randint(0, len(chromosome.gene_sequence))
        
        # Insert the buffer token
        chromosome.gene_sequence.insert(pos, BUFFER_TOKEN)
    
    def simulate_schedule(self, instance: JSSPInstance) -> None:
        """
        Simulate a schedule with uncertainty using an event-driven approach.
        
        This method updates the instance's schedule based on:
        1. Job arrival times
        2. Variable processing times
        3. Machine breakdowns
        
        Args:
            instance: The JSSP instance to simulate
        """
        # Initialize the event queue (priority queue)
        event_queue = []
        
        # Track the current status of jobs and machines
        job_next_op = [0] * len(instance.jobs)
        machine_available = [True] * len(instance.machines)
        current_time = 0
        
        # Add job arrival events to the queue
        for j, job in enumerate(instance.jobs):
            heapq.heappush(event_queue, Event(
                event_type=EventType.JOB_ARRIVAL,
                time=job.arrival_time,
                job_id=j
            ))
        
        # Add machine breakdown events
        for m, machine in enumerate(instance.machines):
            for start_time, end_time in machine.breakdown_times:
                heapq.heappush(event_queue, Event(
                    event_type=EventType.MACHINE_DOWN,
                    time=start_time,
                    machine_id=m,
                    duration=end_time - start_time
                ))
        
        # Process events until queue is empty
        while event_queue:
            event = heapq.heappop(event_queue)
            current_time = event.time
            
            if event.event_type == EventType.JOB_ARRIVAL:
                # A job has arrived - try to schedule its first operation
                job_id = event.job_id
                job = instance.jobs[job_id]
                
                # If the job has operations to process
                if job_next_op[job_id] < len(job.operations):
                    op_idx = job_next_op[job_id]
                    operation = job.operations[op_idx]
                    machine_id = operation.machine_id
                    
                    # If the machine is available, start the operation
                    if machine_available[machine_id]:
                        # Schedule the operation
                        start_time = current_time
                        duration = operation.processing_time
                        
                        operation.start_time = start_time
                        operation.end_time = start_time + duration
                        
                        # Update machine status
                        machine_available[machine_id] = False
                        
                        # Create operation end event
                        heapq.heappush(event_queue, Event(
                            event_type=EventType.OPERATION_END,
                            time=start_time + duration,
                            job_id=job_id,
                            machine_id=machine_id,
                            operation_id=op_idx
                        ))
                        
                        # Add this operation to the machine's scheduled operations
                        instance.machines[machine_id].schedule_operation(job, operation, start_time)
            
            elif event.event_type == EventType.OPERATION_END:
                # An operation has finished
                job_id = event.job_id
                machine_id = event.machine_id
                
                # Make the machine available again
                machine_available[machine_id] = True
                
                # Move to the next operation in this job
                job_next_op[job_id] += 1
                
                # If the job has more operations, create a job arrival event for the next operation
                if job_next_op[job_id] < len(instance.jobs[job_id].operations):
                    heapq.heappush(event_queue, Event(
                        event_type=EventType.JOB_ARRIVAL,
                        time=current_time,  # The next operation is available immediately
                        job_id=job_id
                    ))
                    
                # Check if there are waiting operations for this machine
                waiting_operations = []
                for j, job in enumerate(instance.jobs):
                    if job_next_op[j] < len(job.operations):
                        next_op = job.operations[job_next_op[j]]
                        if next_op.machine_id == machine_id:
                            waiting_operations.append((j, job_next_op[j]))
                
                # If there are waiting operations, schedule the next one
                if waiting_operations:
                    # Here we could use different dispatching rules
                    # For now, just take the first one (FIFO)
                    next_job_id, next_op_idx = waiting_operations[0]
                    
                    # Create a job arrival event for this operation
                    heapq.heappush(event_queue, Event(
                        event_type=EventType.JOB_ARRIVAL,
                        time=current_time,
                        job_id=next_job_id
                    ))
            
            elif event.event_type == EventType.MACHINE_DOWN:
                # Machine breakdown event
                machine_id = event.machine_id
                
                # Mark the machine as unavailable
                machine_available[machine_id] = False
                
                # If there was an operation in progress, interrupt it
                for j, job in enumerate(instance.jobs):
                    for op_idx, op in enumerate(job.operations):
                        if (op.machine_id == machine_id and 
                            op.is_scheduled() and 
                            op.start_time <= current_time < op.end_time):
                            
                            # Calculate remaining time
                            remaining = op.end_time - current_time
                            op.remaining_time = remaining
                            
                            # Update end time to the breakdown time
                            op.end_time = current_time
                            
                            # We'll reschedule this operation when the machine is up again
                            # Create a job arrival event for after repair
                            heapq.heappush(event_queue, Event(
                                event_type=EventType.JOB_ARRIVAL,
                                time=current_time + event.duration,  # After repair
                                job_id=j
                            ))
                
                # Schedule machine up event
                heapq.heappush(event_queue, Event(
                    event_type=EventType.MACHINE_UP,
                    time=current_time + event.duration,
                    machine_id=machine_id
                ))
            
            elif event.event_type == EventType.MACHINE_UP:
                # Machine is repaired and available again
                machine_id = event.machine_id
                machine_available[machine_id] = True
                
                # Check if there are waiting operations for this machine
                waiting_operations = []
                for j, job in enumerate(instance.jobs):
                    if job_next_op[j] < len(job.operations):
                        next_op = job.operations[job_next_op[j]]
                        if next_op.machine_id == machine_id:
                            waiting_operations.append((j, job_next_op[j]))
                
                # If there are waiting operations, schedule the next one
                if waiting_operations:
                    # Create a job arrival event for the first waiting operation
                    next_job_id, _ = waiting_operations[0]
                    heapq.heappush(event_queue, Event(
                        event_type=EventType.JOB_ARRIVAL,
                        time=current_time,
                        job_id=next_job_id
                    ))
            
            elif event.event_type == EventType.BUFFER_END:
                # A buffer period has ended, nothing special to do
                pass 