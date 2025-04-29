"""
Simple scheduling heuristics for Job Shop Scheduling Problems.
"""

import time
from typing import Dict, List, Any, Tuple

from jssp.data import JSSPInstance, Job, Operation
from jssp.schedulers import Scheduler


class FIFOScheduler(Scheduler):
    """
    First-In-First-Out (FIFO) scheduler.
    
    Schedules operations in the order they appear in each job's sequence,
    with jobs being prioritized by their order in the instance.
    """
    
    def schedule(self, instance: JSSPInstance) -> Dict[str, Any]:
        start_time = time.time()
        
        # Reset any existing schedule
        instance.reset_schedule()
        
        # Keep track of the current operation index for each job
        current_op_idx = [0] * len(instance.jobs)
        
        # Keep track of the earliest available time for each job
        job_available_times = [0] * len(instance.jobs)
        
        # Continue until all operations are scheduled
        while True:
            # Check if all jobs are completed
            all_scheduled = True
            for j, job in enumerate(instance.jobs):
                if current_op_idx[j] < len(job.operations):
                    all_scheduled = False
                    break
            
            if all_scheduled:
                break
            
            # Find the next operation to schedule (FIFO order)
            selected_job_idx = -1
            for j, job in enumerate(instance.jobs):
                # Skip if this job has no more operations to schedule
                if current_op_idx[j] >= len(job.operations):
                    continue
                
                # Schedule the job that arrived first
                if selected_job_idx == -1 or job.arrival_time < instance.jobs[selected_job_idx].arrival_time:
                    selected_job_idx = j
            
            if selected_job_idx == -1:
                break  # No more operations to schedule
                
            job = instance.jobs[selected_job_idx]
            operation = job.operations[current_op_idx[selected_job_idx]]
            machine = instance.machines[operation.machine_id]
            
            # Calculate the earliest start time for this operation
            # It needs to be after: 
            # 1. The job's previous operation is complete
            # 2. The machine becomes available
            earliest_start_time = max(
                job_available_times[selected_job_idx],
                machine.get_earliest_available_time()
            )
            
            # Schedule the operation
            machine.schedule_operation(job, operation, earliest_start_time)
            
            # Update the job's available time
            job_available_times[selected_job_idx] = operation.end_time
            
            # Move to the next operation for this job
            current_op_idx[selected_job_idx] += 1
        
        end_time = time.time()
        
        return {
            "algorithm": "FIFO",
            "makespan": instance.makespan(),
            "total_flow_time": instance.total_flow_time(),
            "average_flow_time": instance.average_flow_time(),
            "computation_time": end_time - start_time,
            "is_valid": instance.is_valid_schedule(),
        }


class SPTScheduler(Scheduler):
    """
    Shortest Processing Time (SPT) scheduler.
    
    Schedules the operation with the shortest processing time among all available operations.
    """
    
    def schedule(self, instance: JSSPInstance) -> Dict[str, Any]:
        start_time = time.time()
        
        # Reset any existing schedule
        instance.reset_schedule()
        
        # Keep track of the current operation index for each job
        current_op_idx = [0] * len(instance.jobs)
        
        # Keep track of the earliest available time for each job
        job_available_times = [0] * len(instance.jobs)
        
        # Continue until all operations are scheduled
        while True:
            # Collect all available operations
            available_operations: List[Tuple[int, Operation]] = []
            
            for j, job in enumerate(instance.jobs):
                # Skip if this job has no more operations to schedule
                if current_op_idx[j] >= len(job.operations):
                    continue
                
                # The next operation for this job
                operation = job.operations[current_op_idx[j]]
                available_operations.append((j, operation))
            
            if not available_operations:
                break  # No more operations to schedule
            
            # Sort operations by processing time (SPT rule)
            available_operations.sort(key=lambda x: x[1].processing_time)
            
            # Select the operation with the shortest processing time
            selected_job_idx, operation = available_operations[0]
            job = instance.jobs[selected_job_idx]
            machine = instance.machines[operation.machine_id]
            
            # Calculate the earliest start time for this operation
            earliest_start_time = max(
                job_available_times[selected_job_idx],
                machine.get_earliest_available_time()
            )
            
            # Schedule the operation
            machine.schedule_operation(job, operation, earliest_start_time)
            
            # Update the job's available time
            job_available_times[selected_job_idx] = operation.end_time
            
            # Move to the next operation for this job
            current_op_idx[selected_job_idx] += 1
        
        end_time = time.time()
        
        return {
            "algorithm": "SPT",
            "makespan": instance.makespan(),
            "total_flow_time": instance.total_flow_time(),
            "average_flow_time": instance.average_flow_time(),
            "computation_time": end_time - start_time,
            "is_valid": instance.is_valid_schedule(),
        } 