import uuid
import random
import numpy as np
from typing import List, Dict, Tuple, Optional

class Operation:
    """Represents a single operation in a job that needs to be processed on a specific machine."""
    
    def __init__(self, operation_id: int, machine_id: int, processing_time: int):
        self.operation_id = operation_id  # Position in the job sequence
        self.machine_id = machine_id
        self.nominal_processing_time = processing_time  # Original processing time
        self.processing_time = processing_time  # Actual processing time (may include variability)
        self.start_time: Optional[int] = None
        self.end_time: Optional[int] = None
        self.remaining_time: Optional[int] = None  # For operations interrupted by breakdowns
        
    def apply_variability(self, delta: float) -> None:
        """
        Apply processing time variability.
        
        Args:
            delta: Maximum proportion of variability (e.g., 0.2 for ±20%)
        """
        noise = random.uniform(-delta, delta)
        self.processing_time = max(1, int(self.nominal_processing_time * (1.0 + noise)))
        
    def is_scheduled(self) -> bool:
        """Check if the operation has been scheduled."""
        return self.start_time is not None and self.end_time is not None
    
    def __repr__(self) -> str:
        status = f"[{self.start_time}-{self.end_time}]" if self.is_scheduled() else "[Not scheduled]"
        return f"Op(m{self.machine_id}, {self.processing_time}t) {status}"

class Job:
    """Represents a job consisting of a sequence of operations."""
    
    def __init__(self, job_id: int):
        self.job_id = job_id
        self.operations: List[Operation] = []
        self.arrival_time = 0  # When the job enters the system
    
    def add_operation(self, machine_id: int, processing_time: int) -> None:
        """Adds an operation to this job's sequence."""
        operation_id = len(self.operations)
        self.operations.append(Operation(operation_id, machine_id, processing_time))
    
    def set_arrival_time(self, arrival_time: float) -> None:
        """Set the arrival time for this job."""
        self.arrival_time = arrival_time
    
    def total_processing_time(self) -> int:
        """Calculate the total processing time for all operations in this job."""
        return sum(op.processing_time for op in self.operations)
    
    def is_completed(self) -> bool:
        """Check if all operations in the job have been scheduled."""
        return all(op.is_scheduled() for op in self.operations)
    
    def completion_time(self) -> Optional[int]:
        """Get the completion time of the job (end time of the last operation)."""
        if not self.is_completed():
            return None
        return max(op.end_time for op in self.operations)
    
    def __repr__(self) -> str:
        return f"Job {self.job_id}: {len(self.operations)} operations, total time: {self.total_processing_time()}"

class Machine:
    """Represents a machine that processes operations."""
    
    def __init__(self, machine_id: int):
        self.machine_id = machine_id
        self.scheduled_operations: List[Tuple[Job, Operation]] = []
        self.breakdown_times: List[Tuple[float, float]] = []  # List of (start_time, end_time) for breakdowns
        self.available = True  # Current availability status
        self.failure_rate = 0.0  # Failures per time unit
        self.min_repair_time = 0  # Minimum repair time
        self.max_repair_time = 0  # Maximum repair time
    
    def schedule_operation(self, start: int, end: int, op_id: Tuple[int, int]):
        """Schedule <op_id> if the window [start,end) is free and breakdown-free."""
        # 1) overlap check
        if any(not (end <= s or start >= e) for s, e, _ in self.scheduled_operations):
            raise ValueError(f"Overlap on machine {self.machine_id} for op {op_id}")
        # 2) breakdown check
        if any(b_start < end and start < b_end for b_start, b_end in self.breakdown_times):
            raise ValueError(f"Breakdown clash on machine {self.machine_id} for op {op_id}")
        # OK – commit
        self.scheduled_operations.append((start, end, op_id))
        self.next_free = max(self.next_free, end)
    
    def set_failure_parameters(self, failure_rate: float, min_repair_time: float, max_repair_time: float) -> None:
        """Set the parameters for machine failures."""
        self.failure_rate = failure_rate
        self.min_repair_time = min_repair_time
        self.max_repair_time = max_repair_time
    
    def add_breakdown(self, start_time: float, duration: float) -> None:
        """Add a breakdown period for this machine."""
        end_time = start_time + duration
        self.breakdown_times.append((start_time, end_time))
        self.breakdown_times.sort()  # Sort by start time
    
    def is_in_breakdown(self, time: float) -> bool:
        """Check if the machine is in breakdown at the given time."""
        for start, end in self.breakdown_times:
            if start <= time < end:
                return True
        return False
    
    def get_earliest_available_time(self, start_from: int = 0) -> int:
        """
        First free instant ≥ start_from that is *not* inside a breakdown window
        and does not overlap any scheduled operation.
        """
        t = max(start_from, self.next_free)          # self.next_free is kept by schedule_operation
        while True:
            # skip breakdowns that cover t
            for b_start, b_end in self.breakdown_times:
                if b_start <= t < b_end:
                    t = b_end
                    break
            else:
                # ensure no op overlaps (gaps are OK)
                clash = next(((s, e) for s, e, _ in self.scheduled_operations if s <= t < e), None)
                if clash:
                    t = clash[1]                     # jump to end of that op
                else:
                    return t
    
    def is_available(self, time: int) -> bool:
        """Check if the machine is available at the given time."""
        # Check if the machine is in breakdown
        if self.is_in_breakdown(time):
            return False
            
        # Check if there's an operation scheduled at this time
        for _, op in self.scheduled_operations:
            if op.start_time <= time < op.end_time:
                return False
        return True
    
    def get_next_available_time(self, time: int) -> int:
        """Get the next time the machine becomes available after the given time."""
        # Get the end times of operations that end after the given time
        op_end_times = [op.end_time for _, op in self.scheduled_operations 
                       if op.end_time > time]
        
        # Get the end times of breakdowns that end after the given time
        breakdown_end_times = [end for start, end in self.breakdown_times 
                             if end > time]
        
        available_times = op_end_times + breakdown_end_times
        return min(available_times) if available_times else time
    
    def __repr__(self) -> str:
        return f"Machine {self.machine_id}: {len(self.scheduled_operations)} scheduled operations"

class JSSPInstance:
    """Represents a complete Job Shop Scheduling Problem instance."""
    
    def __init__(self, num_jobs: int = 0, num_machines: int = 0):
        self.jobs: List[Job] = []
        self.machines: List[Machine] = []
        self.processing_time_variability: float = 0.0  # Delta for processing time variability
        
        # Initialize machines
        for i in range(num_machines):
            self.machines.append(Machine(i))
            
        # Initialize jobs (without operations yet)
        for i in range(num_jobs):
            self.jobs.append(Job(i))
    
    def add_job(self, job: Job) -> None:
        """Add a job to the problem instance."""
        self.jobs.append(job)
    
    def add_machine(self, machine: Machine) -> None:
        """Add a machine to the problem instance."""
        self.machines.append(machine)
    
    def set_processing_time_variability(self, delta: float) -> None:
        """Set the processing time variability parameter."""
        self.processing_time_variability = delta
        
        # Apply variability to all existing operations
        for job in self.jobs:
            for op in job.operations:
                op.apply_variability(delta)
    
    def generate_job_arrival_times(self, method: str = "exponential", lambda_arrival: float = 0.1, max_arrival_window: float = 100) -> None:
        """
        Generate arrival times for all jobs.
        
        Args:
            method: Either 'exponential' or 'uniform'
            lambda_arrival: Parameter for exponential distribution (only used if method='exponential')
            max_arrival_window: Maximum arrival time (only used if method='uniform')
        """
        for i, job in enumerate(self.jobs):
            if i == 0:  # First job arrives at time 0
                job.arrival_time = 0
            else:
                if method == "exponential":
                    # Exponential distribution
                    job.arrival_time = self.jobs[i-1].arrival_time + random.expovariate(lambda_arrival)
                else:
                    # Uniform distribution
                    job.arrival_time = random.uniform(0, max_arrival_window)
    
    def setup_machine_breakdowns(self, lambda_fail: float, min_repair: float, max_repair: float) -> None:
        """
        Set up parameters for machine breakdowns.
        
        Args:
            lambda_fail: Failure rate per time unit
            min_repair: Minimum repair time
            max_repair: Maximum repair time
        """
        for machine in self.machines:
            machine.set_failure_parameters(lambda_fail, min_repair, max_repair)
    
    def generate_machine_breakdowns(self, simulation_horizon: int) -> None:
        """
        Generate breakdown events for machines up to a simulation horizon.
        
        Args:
            simulation_horizon: The time limit for simulation
        """
        for machine in self.machines:
            if machine.failure_rate <= 0:
                continue  # Skip machines with no failures
                
            current_time = 0
            while current_time < simulation_horizon:
                # Time to next failure
                if machine.failure_rate > 0:
                    time_to_fail = random.expovariate(machine.failure_rate)
                else:
                    break
                    
                failure_start = current_time + time_to_fail
                if failure_start >= simulation_horizon:
                    break
                    
                # Duration of repair
                repair_time = random.uniform(machine.min_repair_time, machine.max_repair_time)
                
                # Add breakdown
                machine.add_breakdown(failure_start, repair_time)
                
                # Update current time
                current_time = failure_start + repair_time
    
    def reset_schedule(self) -> None:
        """Reset all scheduling information."""
        # Reset operations in jobs
        for job in self.jobs:
            for op in job.operations:
                op.start_time = None
                op.end_time = None
                op.remaining_time = None
        
        # Reset machines
        for machine in self.machines:
            machine.scheduled_operations = []
    
    def is_valid_schedule(self) -> bool:
        """Check if the current schedule is valid."""
        # Check each machine for overlapping operations
        for machine in self.machines:
            sorted_ops = sorted(machine.scheduled_operations, key=lambda x: x[1].start_time)
            for i in range(1, len(sorted_ops)):
                prev_op = sorted_ops[i-1][1]
                curr_op = sorted_ops[i][1]
                if prev_op.end_time > curr_op.start_time:
                    return False
        
        # Check each job for operation sequence (precedence constraints)
        for job in self.jobs:
            for i in range(1, len(job.operations)):
                prev_op = job.operations[i-1]
                curr_op = job.operations[i]
                if not (prev_op.is_scheduled() and curr_op.is_scheduled()):
                    continue
                if prev_op.end_time > curr_op.start_time:
                    return False
                    
        # Check that operations don't overlap with machine breakdowns
        for machine in self.machines:
            for _, op in machine.scheduled_operations:
                for breakdown_start, breakdown_end in machine.breakdown_times:
                    # Check if operation overlaps with any breakdown
                    if (op.start_time < breakdown_end and op.end_time > breakdown_start):
                        return False
        
        return True
    
    def makespan(self) -> int:
        """Calculate the makespan (total completion time) of the current schedule."""
        max_end_time = 0
        for job in self.jobs:
            for op in job.operations:
                if op.is_scheduled() and op.end_time > max_end_time:
                    max_end_time = op.end_time
        return max_end_time
    
    def total_flow_time(self) -> int:
        """Calculate the total flow time (sum of all job completion times)."""
        total = 0
        for job in self.jobs:
            completion_time = job.completion_time()
            if completion_time is not None:
                # Flow time is completion time minus arrival time
                total += (completion_time - job.arrival_time)
        return total
    
    def average_flow_time(self) -> float:
        """Calculate the average flow time across all jobs."""
        return self.total_flow_time() / len(self.jobs) if self.jobs else 0
    
    def copy(self) -> 'JSSPInstance':
        """Create a deep copy of this instance."""
        # Create a new instance with the same number of machines
        instance_copy = JSSPInstance(0, len(self.machines))
        
        # Copy processing time variability
        instance_copy.processing_time_variability = self.processing_time_variability
        
        # Clear the automatically created jobs and machines
        instance_copy.jobs = []
        instance_copy.machines = []
        
        # Copy machines
        for machine in self.machines:
            machine_copy = Machine(machine.machine_id)
            machine_copy.breakdown_times = list(machine.breakdown_times)
            machine_copy.failure_rate = machine.failure_rate
            machine_copy.min_repair_time = machine.min_repair_time
            machine_copy.max_repair_time = machine.max_repair_time
            machine_copy.available = machine.available
            instance_copy.machines.append(machine_copy)
        
        # Copy jobs and operations
        for job in self.jobs:
            job_copy = Job(job.job_id)
            job_copy.arrival_time = job.arrival_time
            
            # Copy operations
            for op in job.operations:
                job_copy.add_operation(op.machine_id, op.nominal_processing_time)
                # Copy the actual processing time with variability
                job_copy.operations[-1].processing_time = op.processing_time
                # Copy scheduling information if available
                if op.is_scheduled():
                    job_copy.operations[-1].start_time = op.start_time
                    job_copy.operations[-1].end_time = op.end_time
                    job_copy.operations[-1].remaining_time = op.remaining_time
            
            instance_copy.jobs.append(job_copy)
        
        # Recreate the scheduled_operations lists in machines
        for m_idx, machine in enumerate(self.machines):
            machine_copy = instance_copy.machines[m_idx]
            
            for job, op in machine.scheduled_operations:
                # Find corresponding job and operation in the copy
                job_copy = next(j for j in instance_copy.jobs if j.job_id == job.job_id)
                op_copy = job_copy.operations[op.operation_id]
                
                # Add to scheduled operations
                machine_copy.scheduled_operations.append((job_copy, op_copy))
        
        return instance_copy
    
    @classmethod
    def generate_random_instance(cls, num_jobs: int, num_machines: int, 
                               min_proc_time: int = 1, max_proc_time: int = 10,
                               seed: Optional[int] = None,
                               arrival_time_method: str = None,
                               lambda_arrival: float = 0.1,
                               max_arrival_window: float = 100,
                               proc_time_variability: float = 0.0,
                               machine_failure_rate: float = 0.0,
                               min_repair_time: float = 5,
                               max_repair_time: float = 20,
                               simulation_horizon: int = 1000) -> 'JSSPInstance':
        """
        Generate a random JSSP instance.
        
        Args:
            num_jobs: Number of jobs
            num_machines: Number of machines
            min_proc_time: Minimum processing time
            max_proc_time: Maximum processing time
            seed: Random seed for reproducibility
            arrival_time_method: Method for generating arrival times ('exponential', 'uniform', or None)
            lambda_arrival: Parameter for exponential arrival distribution
            max_arrival_window: Maximum time window for uniform arrival distribution
            proc_time_variability: Delta for processing time variability (0.0 = no variability)
            machine_failure_rate: Rate of machine failures per time unit (0.0 = no failures)
            min_repair_time: Minimum repair time for machines
            max_repair_time: Maximum repair time for machines
            simulation_horizon: Time horizon for generating machine breakdowns
            
        Returns:
            A new JSSP instance with the specified parameters
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        instance = cls(num_jobs, num_machines)
        
        # For each job, create operations for each machine in random order
        for j, job in enumerate(instance.jobs):
            # Create a random permutation of machines for this job
            machine_order = np.random.permutation(num_machines).tolist()
            
            for machine_id in machine_order:
                # Random processing time
                proc_time = random.randint(min_proc_time, max_proc_time)
                job.add_operation(machine_id, proc_time)
        
        # Set up arrival times if specified
        if arrival_time_method:
            instance.generate_job_arrival_times(
                method=arrival_time_method,
                lambda_arrival=lambda_arrival,
                max_arrival_window=max_arrival_window
            )
            
        # Set up processing time variability if specified
        if proc_time_variability > 0:
            instance.set_processing_time_variability(proc_time_variability)
            
        # Set up machine breakdowns if specified
        if machine_failure_rate > 0:
            instance.setup_machine_breakdowns(
                lambda_fail=machine_failure_rate,
                min_repair=min_repair_time,
                max_repair=max_repair_time
            )
            instance.generate_machine_breakdowns(simulation_horizon)
        
        return instance
    
    def __repr__(self) -> str:
        return f"JSSP Instance: {len(self.jobs)} jobs, {len(self.machines)} machines" 