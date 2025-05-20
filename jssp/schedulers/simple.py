"""
Simple scheduling heuristics for Job Shop Scheduling Problems.
"""

import time
from typing import Dict, Any, List, Tuple

from jssp.data import JSSPInstance, Operation
from jssp.schedulers import Scheduler


class FIFOScheduler(Scheduler):
    """
    First-In-First-Out (FIFO) scheduler.
    Schedules operations in the order they appear in each job's sequence,
    with jobs being prioritized by their order in the instance.
    """

    def schedule(self, instance: JSSPInstance) -> Dict[str, Any]:
        # Measure runtime
        start_time = time.time()

        # Reset any existing schedule
        instance.reset_schedule()

        num_jobs = len(instance.jobs)
        current_op_idx = [0] * num_jobs
        job_available_times = [0.0] * num_jobs
        machine_available_times = [0.0] * instance.num_machines

        # Continue until all operations are scheduled
        while True:
            # Select next job with ops remaining
            selected = [i for i in range(num_jobs) if current_op_idx[i] < len(instance.jobs[i].operations)]
            if not selected:
                break
            # FIFO: pick smallest job index (instance order)
            job_idx = min(selected)
            op = instance.jobs[job_idx].operations[current_op_idx[job_idx]]

            # Determine start timestamp
            ready = max(job_available_times[job_idx], machine_available_times[op.machine_id])
            op.start_time = ready
            op.end_time = ready + op.processing_time
            instance.add_scheduled_operation(job_idx, current_op_idx[job_idx], op.start_time)

            # Update availability
            job_available_times[job_idx] = op.end_time
            machine_available_times[op.machine_id] = op.end_time
            current_op_idx[job_idx] += 1

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
    Schedules the operation with the shortest processing time among all ready operations.
    """

    def schedule(self, instance: JSSPInstance) -> Dict[str, Any]:
        # Measure runtime
        start_time = time.time()

        # Reset any existing schedule
        instance.reset_schedule()

        num_jobs = len(instance.jobs)
        current_op_idx = [0] * num_jobs
        job_available_times = [0.0] * num_jobs
        machine_available_times = [0.0] * instance.num_machines

        while True:
            # Identify all currently ready operations
            ready_ops: List[Tuple[int, Operation]] = []
            for j in range(num_jobs):
                if current_op_idx[j] < len(instance.jobs[j].operations):
                    op = instance.jobs[j].operations[current_op_idx[j]]
                    if job_available_times[j] <= machine_available_times[op.machine_id]:
                        ready_ops.append((j, op))
            if not ready_ops:
                # Either finished or waiting on resources
                # Check if all scheduled
                if all(idx >= len(instance.jobs[j].operations) for j, idx in enumerate(current_op_idx)):
                    break
                # Otherwise, advance time to next machine release
                try:
                    job_min_time = min(job_available_times)
                    # Filter machine times that are greater than minimum job time
                    later_machine_times = [mt for mt in machine_available_times if mt > job_min_time]
                    
                    if not later_machine_times:
                        # If no machine times are later than the min job time, just use the min job time
                        next_machine_time = job_min_time
                    else:
                        next_machine_time = min(later_machine_times)
                        
                    for j in range(num_jobs):
                        job_available_times[j] = max(job_available_times[j], next_machine_time)
                except ValueError:
                    # If we hit a value error (empty list), we're done
                    break
                continue

            # Pick the ready operation with smallest processing time
            job_idx, op = min(ready_ops, key=lambda x: x[1].processing_time)
            start = max(job_available_times[job_idx], machine_available_times[op.machine_id])
            op.start_time = start
            op.end_time = start + op.processing_time
            instance.add_scheduled_operation(job_idx, current_op_idx[job_idx], op.start_time)

            job_available_times[job_idx] = op.end_time
            machine_available_times[op.machine_id] = op.end_time
            current_op_idx[job_idx] += 1

        end_time = time.time()
        return {
            "algorithm": "SPT",
            "makespan": instance.makespan(),
            "total_flow_time": instance.total_flow_time(),
            "average_flow_time": instance.average_flow_time(),
            "computation_time": end_time - start_time,
            "is_valid": instance.is_valid_schedule(),
        }
