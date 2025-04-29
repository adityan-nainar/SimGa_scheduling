"""
Simple test script for the JSSP framework.
"""

from jssp.data import JSSPInstance
from jssp.schedulers.simple import FIFOScheduler, SPTScheduler
from jssp.schedulers.genetic import GeneticScheduler
import matplotlib.pyplot as plt

# Generate a simple instance
instance = JSSPInstance.generate_random_instance(
    num_jobs=3,
    num_machines=2,
    min_proc_time=1,
    max_proc_time=10,
    seed=42
)

print("Generated instance:")
print(instance)

for job in instance.jobs:
    print(f"\nJob {job.job_id}:")
    for op in job.operations:
        print(f"  Operation on machine {op.machine_id}: {op.processing_time} time units")

# Test FIFO scheduler
print("\n=== Testing FIFO Scheduler ===")
fifo = FIFOScheduler()
fifo_result = fifo.schedule(instance.copy())
print(f"Makespan: {fifo_result['makespan']}")
print(f"Valid schedule: {fifo_result['is_valid']}")

# Test SPT scheduler
print("\n=== Testing SPT Scheduler ===")
spt = SPTScheduler()
spt_result = spt.schedule(instance.copy())
print(f"Makespan: {spt_result['makespan']}")
print(f"Valid schedule: {spt_result['is_valid']}")

# Test GA scheduler
print("\n=== Testing Genetic Algorithm Scheduler ===")
ga = GeneticScheduler(population_size=20, generations=50)
ga_result = ga.schedule(instance.copy())
print(f"Makespan: {ga_result['makespan']}")
print(f"Valid schedule: {ga_result['is_valid']}")
print(f"Best fitness history: {ga_result['best_fitness_history']}")

print("\nAll tests completed successfully!") 