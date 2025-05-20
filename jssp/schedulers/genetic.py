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
        print("      Starting chromosome decode_and_evaluate")
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
        print(f"      Decoding {len(self.gene_sequence)} genes")
        gene_count = 0
        for gene in self.gene_sequence:
            gene_count += 1
            if gene_count % 100 == 0:
                print(f"        Decoded {gene_count}/{len(self.gene_sequence)} genes")
                
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
        print("      Calculating fitness")
        makespan = self.instance.makespan()
        
        # Check if schedule is valid
        is_valid = self.instance.is_valid_schedule()
        
        # Apply penalty for invalid schedules
        if not is_valid:
            print("      Warning: Invalid schedule found, applying penalty")
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
        print(f"      Decode complete. Makespan: {makespan}, valid: {is_valid}")
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
    
    def schedule(self, instance: JSSPInstance, max_time_seconds: int = 60) -> Dict[str, Any]:
        print("Starting GeneticScheduler.schedule method")
        start_time = time.time()
        max_end_time = start_time + max_time_seconds
        
        # Create initial population
        print("Creating initial population")
        population = self._initialize_population(instance)
        
        # Fitness cache to avoid recomputing fitness for identical chromosomes
        fitness_cache = {}
        
        # Evaluate the initial population
        print(f"Evaluating initial population of size {len(population)}")
        chromosome_count = 0
        for chromosome in population:
            chromosome_count += 1
            if chromosome_count % 10 == 0:
                print(f"  Evaluated {chromosome_count}/{len(population)} chromosomes")
                
            # Check if we've evaluated an identical chromosome already
            gene_tuple = tuple(chromosome.gene_sequence)
            if gene_tuple in fitness_cache:
                chromosome.fitness = fitness_cache[gene_tuple]["fitness"]
                chromosome.schedule_result = fitness_cache[gene_tuple]["result"]
            else:
                print(f"  Decoding and evaluating chromosome {chromosome_count}")
                chromosome.decode_and_evaluate()
                
                # If using uncertainty, evaluate with multiple simulations
                if self.use_uncertainty:
                    print(f"  Evaluating chromosome {chromosome_count} with uncertainty")
                    # We need to copy the instance since evaluate_with_uncertainty will modify it
                    instance_copy = chromosome.instance.copy()
                    uncertainty_results = self.evaluate_with_uncertainty(instance_copy)
                    chromosome.fitness = uncertainty_results["robust_fitness"]
                    chromosome.schedule_result.update(uncertainty_results)
                
                # Cache the result
                fitness_cache[gene_tuple] = {
                    "fitness": chromosome.fitness,
                    "result": chromosome.schedule_result
                }
                
            # Check if we've exceeded the time limit
            if time.time() > max_end_time:
                print(f"Warning: Initial population evaluation exceeded time limit of {max_time_seconds} seconds.")
                break
        
        # Sort by fitness (makespan - lower is better)
        population.sort(key=lambda x: x.fitness)
        
        best_fitness_history = [population[0].fitness]
        avg_fitness_history = [sum(c.fitness for c in population) / len(population)]
        
        print(f"Initial population evaluation complete. Best fitness: {population[0].fitness}")
        
        # Main evolutionary loop
        for generation in range(self.generations):
            print(f"Starting generation {generation+1}/{self.generations}")
            # Check if we've exceeded the time limit
            if time.time() > max_end_time:
                print(f"Warning: GA terminated after {generation} generations due to time limit of {max_time_seconds} seconds.")
                break
                
            # Create new population with elitism
            new_population = population[:self.elitism_count]
            
            # Fill the rest of the population
            offspring_count = 0
            while len(new_population) < self.population_size:
                offspring_count += 1
                if offspring_count % 10 == 0:
                    print(f"  Created {offspring_count} offspring in generation {generation+1}")
                    
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
                
                # Check if we've exceeded the time limit
                if time.time() > max_end_time:
                    break
            
            # Replace old population
            population = new_population
            
            # Evaluate the new population
            print(f"  Evaluating population for generation {generation+1}")
            chromosome_count = 0
            for chromosome in population:
                chromosome_count += 1
                if chromosome_count % 10 == 0:
                    print(f"    Evaluated {chromosome_count}/{len(population)} chromosomes in generation {generation+1}")
                    
                # Check if we've evaluated an identical chromosome already
                gene_tuple = tuple(chromosome.gene_sequence)
                if gene_tuple in fitness_cache:
                    chromosome.fitness = fitness_cache[gene_tuple]["fitness"]
                    chromosome.schedule_result = fitness_cache[gene_tuple]["result"]
                elif chromosome.fitness == float('inf'):  # Only evaluate if not already evaluated
                    chromosome.decode_and_evaluate()
                    
                    # If using uncertainty, evaluate with multiple simulations
                    if self.use_uncertainty:
                        instance_copy = chromosome.instance.copy()
                        uncertainty_results = self.evaluate_with_uncertainty(instance_copy)
                        chromosome.fitness = uncertainty_results["robust_fitness"]
                        chromosome.schedule_result.update(uncertainty_results)
                    
                    # Cache the result
                    fitness_cache[gene_tuple] = {
                        "fitness": chromosome.fitness,
                        "result": chromosome.schedule_result
                    }
                        
                # Check if we've exceeded the time limit
                if time.time() > max_end_time:
                    break
            
            # Sort by fitness
            population.sort(key=lambda x: x.fitness)
            
            # Record history
            best_fitness_history.append(population[0].fitness)
            avg_fitness_history.append(sum(c.fitness for c in population) / len(population))
            
            print(f"  Generation {generation+1} complete. Best fitness: {population[0].fitness}")
            
            # Early stopping if no improvement for 5 generations
            if len(best_fitness_history) > 5 and all(best_fitness_history[-5] == bfh for bfh in best_fitness_history[-5:]):
                print(f"Early stopping at generation {generation+1} due to no improvement in the last 5 generations")
                break
        
        # Return the best solution found
        print("Evolution complete, returning best solution")
        best_solution = population[0]
        best_solution.decode_and_evaluate()  # Make sure the schedule is built
        
        # If using uncertainty, do a final evaluation with more simulations
        if self.use_uncertainty:
            print("Performing final uncertainty evaluation on best solution")
            final_instance = instance.copy()
            # Just evaluate with uncertainty, no need to reconstruct the schedule
            final_results = self.evaluate_with_uncertainty(best_solution.instance.copy())
            best_solution.schedule_result.update(final_results)
        
        result = best_solution.schedule_result.copy()
        result.update({
            "computation_time": time.time() - start_time,
            "best_fitness_history": best_fitness_history,
            "avg_fitness_history": avg_fitness_history,
            "generations": generation + 1 if generation < self.generations else self.generations
        })
        
        print(f"Genetic algorithm completed in {time.time() - start_time:.2f} seconds")
        print(f"Final best fitness: {best_solution.fitness}")
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
        print("  Starting crossover operation")
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
        
        # Build the child sequence with a maximum iteration limit
        max_iterations = len(p1_seq) * 3  # Reasonable limit to prevent infinite loops
        iteration_count = 0
        
        print(f"  Crossover: Building child sequence (p1_len={len(p1_seq)}, p2_len={len(p2_seq)}, max_iter={max_iterations})")
        while len(child_sequence) < len(p1_seq) and iteration_count < max_iterations:
            iteration_count += 1
            if iteration_count % 100 == 0:
                print(f"    Crossover iteration {iteration_count}, child length: {len(child_sequence)}/{len(p1_seq)}")
                
            # Collect all operations that can be scheduled next (where all predecessors are scheduled)
            eligible_ops = []
            for j_id in range(len(instance.jobs)):
                op_idx = available_op_idx[j_id]
                if op_idx < len(instance.jobs[j_id].operations):
                    gene = (j_id, op_idx)
                    if gene not in emitted:
                        eligible_ops.append(gene)
            
            if not eligible_ops:
                # If we've scheduled everything possible but not all operations, 
                # something is wrong with the precedence constraints
                print(f"    No eligible operations at iteration {iteration_count}, breaking loop")
                break
            
            # Use parent preference to choose between eligible operations when possible
            parent_idx = parent_choice[len(child_sequence) % len(parent_choice)]
            parents = [p1_seq, p2_seq]
            
            # Try to find an operation from the preferred parent that is eligible
            selected_gene = None
            for gene in parents[parent_idx]:
                if gene in eligible_ops:
                    selected_gene = gene
                    break
            
            # If no eligible operation from preferred parent, try the other parent
            if selected_gene is None:
                for gene in parents[1 - parent_idx]:
                    if gene in eligible_ops:
                        selected_gene = gene
                        break
            
            # If still no match, just take the first eligible operation
            if selected_gene is None and eligible_ops:
                selected_gene = eligible_ops[0]
            
            # Add the selected gene to the child sequence
            if selected_gene is not None:
                child_sequence.append(selected_gene)
                emitted.add(selected_gene)
                job_id, _ = selected_gene
                available_op_idx[job_id] += 1
            else:
                # No eligible operations but we haven't scheduled everything
                # This is a fail-safe to avoid infinite loops
                print(f"    No gene selected at iteration {iteration_count}, breaking loop")
                break
        
        # If we didn't schedule all operations due to a potential loop,
        # fall back to a simple topological sort
        if len(child_sequence) < len(p1_seq):
            print(f"  Crossover: Incomplete child sequence ({len(child_sequence)}/{len(p1_seq)}), using fallback")
            # Reset available operations
            available_op_idx = [0] * len(instance.jobs)
            child_sequence = []
            emitted = set()
            
            # Create a topological sort of operations
            for i in range(len(p1_seq)):
                if i % 100 == 0:
                    print(f"    Fallback iteration {i}/{len(p1_seq)}")
                fallback_applied = False
                for j_id in range(len(instance.jobs)):
                    op_idx = available_op_idx[j_id]
                    if op_idx < len(instance.jobs[j_id].operations):
                        gene = (j_id, op_idx)
                        if gene not in emitted:
                            child_sequence.append(gene)
                            emitted.add(gene)
                            available_op_idx[j_id] += 1
                            fallback_applied = True
                            break
                if not fallback_applied:
                    print(f"    Fallback method failed at iteration {i}")
                    break
        
        print(f"  Crossover: Completed with final child sequence length {len(child_sequence)}")
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


    def simulate_schedule(self, instance: JSSPInstance, max_events: int = 1000) -> None:
        """
        Simulate a schedule with uncertainty using an event-driven approach.
        
        This method updates the instance's schedule based on:
        1. Job arrival times
        2. Variable processing times
        3. Machine breakdowns
        
        Args:
            instance: The JSSP instance to simulate
            max_events: Maximum number of events to process (prevents infinite loops)
        """
        print("      Starting simulate_schedule method")
        
        # If uncertainty is disabled, skip simulation entirely
        if not self.use_uncertainty:
            print("      Uncertainty disabled, skipping simulation")
            return
            
        # Initialize the event queue (priority queue)
        event_queue = []
        
        # Track the current status of jobs and machines
        job_next_op = [0] * len(instance.jobs)
        machine_available = [True] * len(instance.machines)
        current_time = 0
        
        # Add job arrival events to the queue
        arrival_events = 0
        for j, job in enumerate(instance.jobs):
            if job.arrival_time > 0:  # Only add if arrival time is non-zero
                heapq.heappush(event_queue, Event(
                    event_type=EventType.JOB_ARRIVAL,
                    time=job.arrival_time,
                    job_id=j
                ))
                arrival_events += 1
            else:
                # For jobs arriving at time 0, process them directly
                if job_next_op[j] < len(job.operations):
                    operation = job.operations[job_next_op[j]]
                    machine_id = operation.machine_id
                    if machine_available[machine_id]:
                        # Schedule this operation
                        operation.start_time = 0
                        operation.end_time = operation.processing_time
                        machine_available[machine_id] = False
                        
                        # Add end event
                        heapq.heappush(event_queue, Event(
                            event_type=EventType.OPERATION_END,
                            time=operation.end_time,
                            job_id=j,
                            machine_id=machine_id,
                            operation_id=job_next_op[j]
                        ))
                        
                        # Add to scheduled operations
                        instance.machines[machine_id].schedule_operation_obj(job, operation)
        
        print(f"      Added {arrival_events} job arrival events to queue")
        
        # Add machine breakdown events - only if machine failure rate is greater than 0
        breakdown_events = 0
        for m, machine in enumerate(instance.machines):
            if machine.failure_rate > 0:
                # Generate breakdowns using exponential distribution
                t = random.expovariate(machine.failure_rate)
                simulation_horizon = instance.makespan() * 2 if instance.makespan() > 0 else 1000  # Estimate based on makespan
                
                # Limit the number of breakdowns per machine
                breakdown_count = 0
                max_breakdowns_per_machine = 5  # Reduced from 10 to improve performance
                
                while t < simulation_horizon and breakdown_count < max_breakdowns_per_machine:
                    # Generate repair time
                    repair_time = random.uniform(machine.min_repair_time, machine.max_repair_time)
                    
                    # Add breakdown events
                    heapq.heappush(event_queue, Event(
                        event_type=EventType.MACHINE_DOWN,
                        time=t,
                        machine_id=m,
                        duration=repair_time
                    ))
                    breakdown_events += 1
                    
                    # Move to next failure time
                    t += random.expovariate(machine.failure_rate) + repair_time
                    breakdown_count += 1
        
        print(f"      Added {breakdown_events} machine breakdown events to queue")
        
        # Process events until queue is empty or max events reached
        events_processed = 0
        max_time = float('inf')  # Keep track of the maximum event time
        
        # Check if all operations are scheduled initially
        all_scheduled = all(
            all(op.is_scheduled() for op in job.operations)
            for job in instance.jobs
        )
        
        # If everything is already scheduled and no machine failures, we can skip simulation
        if all_scheduled and not any(machine.failure_rate > 0 for machine in instance.machines):
            print("      All operations already scheduled and no machine failures, skipping simulation")
            return
            
        print(f"      Starting event simulation with {len(event_queue)} events")
        last_progress_time = time.time()
        while event_queue and events_processed < max_events:
            # Print progress every 5 seconds
            current_progress_time = time.time()
            if current_progress_time - last_progress_time > 5:
                print(f"        Processed {events_processed} events, {len(event_queue)} remaining")
                last_progress_time = current_progress_time
                
            event = heapq.heappop(event_queue)
            current_time = event.time
            events_processed += 1
            
            # Early termination: if the event time exceeds makespan by a large margin,
            # and we've processed a reasonable number of events, we can stop
            if current_time > max_time and events_processed > len(instance.jobs) * len(instance.machines) * 2:
                print(f"        Early termination at event {events_processed}: current_time {current_time} > max_time {max_time}")
                break
                
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
                        
                        # Update max_time
                        max_time = max(max_time, operation.end_time)
                        
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
                if waiting_operations and len(waiting_operations) <= 5:  # Limit checking to 5 operations max for performance
                    # Here we could use different dispatching rules
                    # For now, just take the first one (FIFO)
                    next_job_id, next_op_idx = waiting_operations[0]
                    
                    # Create a job arrival event for this operation
                    heapq.heappush(event_queue, Event(
                        event_type=EventType.JOB_ARRIVAL,
                        time=current_time,
                        job_id=next_job_id
                    ))
            
            # Only include breakdown handling if there are any breakdowns (skip for better performance)
            elif breakdown_events > 0 and event.event_type == EventType.MACHINE_DOWN:
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
                            
                            # Store the current operation to resume later
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
            
            elif breakdown_events > 0 and event.event_type == EventType.MACHINE_UP:
                # Machine is repaired and available again
                machine_id = event.machine_id
                machine_available[machine_id] = True
                
                # Reactivate machine
                instance.machines[machine_id].reactivate(current_time)
                
                # Check if there are waiting operations for this machine
                waiting_operations = []
                for j, job in enumerate(instance.jobs):
                    if job_next_op[j] < len(job.operations):
                        next_op = job.operations[job_next_op[j]]
                        if next_op.machine_id == machine_id:
                            waiting_operations.append((j, job_next_op[j]))
                
                # If there are waiting operations, schedule the next one
                if waiting_operations and len(waiting_operations) <= 5:  # Limit checking to 5 operations max for performance
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
        
        print(f"      Simulation complete. Processed {events_processed} events")
        if events_processed >= max_events:
            print(f"      Warning: Hit maximum event limit ({max_events})")
        
        
        # Update the makespan and other metrics
        makespan = instance.makespan()
        print(f"      Final makespan after simulation: {makespan}")

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
