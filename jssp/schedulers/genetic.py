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
        
        # Create lookup dictionaries for operation processing times and machine ids
        jobs = self.instance.jobs
        ops_lookup = {}
        for job in jobs:
            for i, op in enumerate(job.operations):
                ops_lookup[(job.job_id, i)] = (op.machine_id, op.processing_time)

        # Keep track of current position for each job and machine
        job_cursor = [0] * len(self.instance.jobs)
        machine_cursor = [0] * len(self.instance.machines)
        
        # Track the last processed job and machine for buffer insertion
        current_job = 0
        current_machine = 0
        
        current_machine = current_job = None  # before gene loop

        # Process operations in the order specified by the gene sequence
        for gene in self.gene_sequence:
            if gene == BUFFER_TOKEN:
                if current_machine is None:   # skip if at sequence start
                    continue
                delay = random.randint(1, 5)
                machine_cursor[current_machine] += delay
                job_cursor[current_job]        += delay
                continue

            job_id, op_idx = gene
            machine_id, ptime = ops_lookup[(job_id, op_idx)]
            current_job = job_id
            current_machine = machine_id

            # Honour job arrival time
            earliest_start = max(job_cursor[job_id],
                     machine_cursor[machine_id],
                     jobs[job_id].arrival_time)   # ← add this constraint

            start = earliest_start
            end = start + ptime
            operation = jobs[job_id].operations[op_idx]
            self.instance.machines[machine_id].schedule_operation_obj(jobs[job_id], operation, start)


            job_cursor[job_id] = end
            machine_cursor[machine_id] = end
        
        # Calculate fitness (makespan - lower is better)
        makespan = self.instance.makespan()
        
        # Check if schedule is valid
        is_valid = self.instance.is_valid_schedule()
        
        # Apply penalty for invalid schedules
        if not is_valid:
            makespan += 10_000  # penalty
            
        self.fitness = makespan
        
        result = {
            "algorithm": "GA",
            "makespan": makespan,
            "total_flow_time": self.instance.total_flow_time(),
            "average_flow_time": self.instance.average_flow_time(),
            "is_valid": is_valid,
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
        p1_seq = tuple(gene for gene in parent1.gene_sequence if gene != BUFFER_TOKEN)
        p2_seq = tuple(gene for gene in parent2.gene_sequence if gene != BUFFER_TOKEN)
        
        # Initialize child sequence
        child_sequence = []
        
        # Track available operations for each job
        available_op_idx = [0] * len(instance.jobs)
        
        # Create a list of which parent to choose from at each step
        parent_choice = [random.choice([0, 1]) for _ in range(len(p1_seq))]
        
        # Set to track emitted operations
        emitted = set()
        
        # Function to check if an operation can be emitted
        def can_emit(gene, emitted_set):
            job_id, op_idx = gene
            return op_idx == available_op_idx[job_id] and gene not in emitted_set
        
        # Build the child sequence
        parents = [p1_seq, p2_seq]
        pos = [0, 0]  # Position in each parent sequence

        # Make sure to use a hashable type (tuple) for set operations
        remaining = set(p1_seq)  # All operations that need to be scheduled
        
        while len(child_sequence) < len(p1_seq):
            parent_idx = parent_choice[len(child_sequence)] if len(child_sequence) < len(parent_choice) else random.choice([0, 1])
            
            # Check if this parent has any operations left
            if pos[parent_idx] >= len(parents[parent_idx]):
                # This parent is exhausted, switch to the other one
                parent_idx = 1 - parent_idx
            
            # Check if both parents are exhausted
            if pos[0] >= len(parents[0]) and pos[1] >= len(parents[1]):
                break
                
            # Try to get a valid operation from current parent
            parent_tried = 0
            while parent_tried < 2:  # Try both parents if needed
                current_idx = (parent_idx + parent_tried) % 2
                current_parent = parents[current_idx]
                current_pos = pos[current_idx]
                
                if current_pos >= len(current_parent):
                    parent_tried += 1
                    continue
                    
                gene = current_parent[current_pos]
                # Ensure gene is a tuple
                if isinstance(gene, list):
                    gene = tuple(gene)
                    
                if can_emit(gene, emitted):
                    child_sequence.append(gene)
                    emitted.add(gene)
                    if gene in remaining:  # Check if the gene is in remaining before removing
                        remaining.remove(gene)
                    pos[current_idx] += 1
                    available_op_idx[gene[0]] += 1
                    break
                pos[current_idx] += 1
                parent_tried += 1
            else:  # Fallback: dump *any* remaining eligible gene
                eligible_genes = [g for g in remaining if can_emit(g, emitted)]
                if eligible_genes:
                    gene = eligible_genes[0]
                    child_sequence.append(gene)
                    emitted.add(gene)
                    remaining.remove(gene)
                    available_op_idx[gene[0]] += 1
                elif remaining:  # If there are no eligible genes but there are remaining operations
                    # We need to find a way to make progress - choose any operation where all predecessors are emitted
                    for j_id in range(len(instance.jobs)):
                        op_idx = available_op_idx[j_id]
                        if op_idx < len(instance.jobs[j_id].operations):
                            gene = (j_id, op_idx)
                            child_sequence.append(gene)
                            emitted.add(gene)
                            if gene in remaining:
                                remaining.remove(gene)
                            available_op_idx[j_id] += 1
                            break
        
        # Create and return the child chromosome
        return Chromosome(instance, child_sequence)
    
    def _mutate(self, chromosome: Chromosome) -> None:
        """Swap two genes from *different* jobs."""
        candidates = [(i, j) for i in range(len(chromosome.gene_sequence))
                          for j in range(i + 1, len(chromosome.gene_sequence))
                          if chromosome.gene_sequence[i] != BUFFER_TOKEN 
                          and chromosome.gene_sequence[j] != BUFFER_TOKEN
                          and chromosome.gene_sequence[i][0] != chromosome.gene_sequence[j][0]]
        if candidates:
            i, j = random.choice(candidates)
            chromosome.gene_sequence[i], chromosome.gene_sequence[j] = chromosome.gene_sequence[j], chromosome.gene_sequence[i]
    
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
    

    BUFFER_END   = "BUFFER_END"
    MACHINE_DOWN = "MACHINE_DOWN"
    MACHINE_UP   = "MACHINE_UP"


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
            if machine.failure_rate > 0:
                # Generate breakdowns using exponential distribution
                t = random.expovariate(machine.failure_rate)
                simulation_horizon = instance.makespan() * 2 if instance.makespan() > 0 else 1000  # Estimate based on makespan
                
                while t < simulation_horizon:
                    # Generate repair time
                    repair_time = random.uniform(machine.min_repair_time, machine.max_repair_time)
                    
                    # Add breakdown events
                    heapq.heappush(event_queue, Event(
                        event_type=EventType.MACHINE_DOWN,
                        time=t,
                        machine_id=m,
                        duration=repair_time
                    ))
                    
                    # Move to next failure time
                    t += random.expovariate(machine.failure_rate)
        
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
                        instance.machines[machine_id].schedule_operation_obj(job, operation, start_time)
            
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
                            
                            # NEW - Store the current operation to resume later
                            current_op = op
                            current_op.pause(current_time)  # New method to pause and calculate remaining time
                            
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
                
                # NEW - Reactivate machine
                instance.machines[machine_id].reactivate(current_time)
                
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
                # A buffer period has ended, make the target machine available
                machine_id = event.machine_id
                machine_available[machine_id] = True
                
                # Check for waiting operations
                waiting_operations = []
                for j, job in enumerate(instance.jobs):
                    if job_next_op[j] < len(job.operations):
                        next_op = job.operations[job_next_op[j]]
                        if next_op.machine_id == machine_id:
                            waiting_operations.append((j, job_next_op[j]))
                
                # If there are waiting operations, create an arrival event
                if waiting_operations:
                    next_job_id, _ = waiting_operations[0]
                    heapq.heappush(event_queue, Event(
                        event_type=EventType.JOB_ARRIVAL,
                        time=current_time,
                        job_id=next_job_id
                    ))

    def evaluate(self):
        """Evaluate the chromosome's fitness based on the makespan."""
        makespan = self.instance.makespan()

        if not self.instance.is_valid_schedule():
            makespan += 10_000   # simple static penalty; tune as needed
        self.fitness = makespan
        return makespan
    
def precedence_preserving_crossover(p1: 'Chromosome', p2: 'Chromosome') -> 'Chromosome':
    size, emitted = len(p1.gene_sequence), set()
    pos1 = pos2 = 0
    child = []
    while len(child) < size:
        for parent, pos in ((p1, pos1), (p2, pos2)):
            if len(child) == size:
                break
            while pos < len(parent.gene_sequence):
                gene = parent.gene_sequence[pos]; pos += 1
                # Skip buffer tokens
                if gene == BUFFER_TOKEN:
                    continue
                # Ensure gene is a tuple
                if isinstance(gene, list):
                    gene = tuple(gene)
                job, op = gene
                # Check if all predecessor operations are emitted
                can_emit = True
                for k in range(op):
                    pred_gene = (job, k)
                    if pred_gene not in emitted:
                        can_emit = False
                        break
                if can_emit:
                    child.append(gene); emitted.add(gene); break
            if parent is p1: pos1 = pos
            else:            pos2 = pos
    return Chromosome(p1.instance, child)
