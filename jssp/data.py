import uuid
import random
import numpy as np
from typing import List, Dict, Tuple, Optional

class Operation:
    """Represents a single operation in a job that needs to be processed on a specific machine."""
    
    def __init__(self, operation_id: int, machine_id: int, processing_time: int):
        self.operation_id = operation_id  # Position in the job sequence
        self.machine_id = machine_id
        self.processing_time = processing_time
        self.start_time: Optional[int] = None
        self.end_time: Optional[int] = None
        
    def is_scheduled(self) -> bool:
        """Check if the operation has been scheduled."""
        return self.start_time is not None and self.end_time is not None
    
    def __repr__(self) -> str:
        status = f"[{self.start_time}-{self.end_time}]" if self.is_scheduled() else "[Not scheduled]"
        return f"Op(m{self.machine_id}, {self.processing_time}t) {status}"

class Job:
    """Represents a job consisting of a sequence of operations."""
    
    def __init__(self, job_id: int, delay_probability: float = 0.0):
        self.job_id = job_id
        self.operations: List[Operation] = []
        self.arrival_time = 0  # When the job enters the system
        self.delay_probability = delay_probability  # Probability of job arrival delay
    
    def add_operation(self, machine_id: int, processing_time: int) -> None:
        """Adds an operation to this job's sequence."""
        operation_id = len(self.operations)
        self.operations.append(Operation(operation_id, machine_id, processing_time))
    
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
    
    def is_delayed(self) -> bool:
        """Determine if the job is delayed based on the delay probability."""
        return random.random() < self.delay_probability
    
    def __repr__(self) -> str:
        return f"Job {self.job_id}: {len(self.operations)} operations, total time: {self.total_processing_time()}"

class Machine:
    """Represents a machine that processes operations."""
    
    def __init__(self, machine_id: int, delay_probability: float = 0.0):
        self.machine_id = machine_id
        self.scheduled_operations: List[Tuple[Job, Operation]] = []
        self.delay_probability = delay_probability  # Probability of machine availability delay
    
    def schedule_operation(self, job: Job, operation: Operation, start_time: int) -> None:
        """Schedule an operation on this machine."""
        operation.start_time = start_time
        operation.end_time = start_time + operation.processing_time
        self.scheduled_operations.append((job, operation))
        # Sort operations by start time to maintain timeline
        self.scheduled_operations.sort(key=lambda x: x[1].start_time)
    
    def get_earliest_available_time(self) -> int:
        """Get the earliest time this machine is available for a new operation."""
        if not self.scheduled_operations:
            return 0
        return max(op.end_time for _, op in self.scheduled_operations)
    
    def is_available(self, time: int) -> bool:
        """Check if the machine is available at the given time."""
        for _, op in self.scheduled_operations:
            if op.start_time <= time < op.end_time:
                return False
        return True
    
    def get_next_available_time(self, time: int) -> int:
        """Get the next time the machine becomes available after the given time."""
        available_times = [op.end_time for _, op in self.scheduled_operations 
                         if op.end_time > time]
        return min(available_times) if available_times else time
    
    def is_delayed(self) -> bool:
        """Determine if the machine is delayed based on the delay probability."""
        return random.random() < self.delay_probability
    
    def __repr__(self) -> str:
        return f"Machine {self.machine_id}: {len(self.scheduled_operations)} scheduled operations"

class JSSPInstance:
    """Represents a complete Job Shop Scheduling Problem instance."""
    
    def __init__(self, num_jobs: int = 0, num_machines: int = 0):
        self.jobs: List[Job] = []
        self.machines: List[Machine] = []
        
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
    
    def reset_schedule(self) -> None:
        """Reset all scheduling information."""
        # Reset operations in jobs
        for job in self.jobs:
            for op in job.operations:
                op.start_time = None
                op.end_time = None
        
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
        return sum(job.completion_time() or 0 for job in self.jobs)
    
    def average_flow_time(self) -> float:
        """Calculate the average flow time across all jobs."""
        return self.total_flow_time() / len(self.jobs) if self.jobs else 0
    
    def copy(self) -> 'JSSPInstance':
        """Create a deep copy of this instance."""
        # Create a new instance with the same number of machines
        instance_copy = JSSPInstance(0, len(self.machines))
        
        # Clear the automatically created jobs and machines
        instance_copy.jobs = []
        instance_copy.machines = []
        
        # Copy machines
        for machine in self.machines:
            machine_copy = Machine(machine.machine_id)
            instance_copy.machines.append(machine_copy)
        
        # Copy jobs and operations
        for job in self.jobs:
            job_copy = Job(job.job_id)
            job_copy.arrival_time = job.arrival_time
            
            # Copy operations
            for op in job.operations:
                job_copy.add_operation(op.machine_id, op.processing_time)
                # Copy scheduling information if available
                if op.is_scheduled():
                    job_copy.operations[-1].start_time = op.start_time
                    job_copy.operations[-1].end_time = op.end_time
            
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
                               seed: Optional[int] = None) -> 'JSSPInstance':
        """Generate a random JSSP instance."""
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
        
        return instance
    
    def __repr__(self) -> str:
        return f"JSSP Instance: {len(self.jobs)} jobs, {len(self.machines)} machines" 