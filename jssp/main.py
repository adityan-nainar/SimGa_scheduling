"""
Main module for running JSSP experiments.
"""

import time
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional

from jssp.data import JSSPInstance
from jssp.schedulers.simple import FIFOScheduler, SPTScheduler
from jssp.schedulers.genetic import GeneticScheduler
from jssp.visualization import (
    plot_gantt_chart_matplotlib, 
    plot_comparison_metrics,
    plot_ga_convergence
)


def run_experiment(
    instance: JSSPInstance,
    save_plots: bool = False,
    plots_prefix: str = "experiment",
) -> Dict[str, Any]:
    """
    Run a scheduling experiment with all available algorithms.
    
    Args:
        instance: A JSSP instance to schedule
        save_plots: Whether to save plots to files
        plots_prefix: Prefix for plot filenames
        
    Returns:
        A dictionary with results from all algorithms
    """
    print(f"Running experiment on {instance}")
    
    schedulers = [
        ("FIFO", FIFOScheduler()),
        ("SPT", SPTScheduler()),
        ("GA", GeneticScheduler(population_size=50, generations=100))
    ]
    
    results = {}
    
    for name, scheduler in schedulers:
        print(f"\nRunning {name} scheduler...")
        result = scheduler.schedule(instance)
        print(f"  Makespan: {result['makespan']}")
        print(f"  Total flow time: {result['total_flow_time']}")
        print(f"  Valid schedule: {result['is_valid']}")
        print(f"  Computation time: {result.get('computation_time', 'N/A'):.4f} seconds")
        
        results[name] = result
        
        # Plot Gantt chart
        if save_plots:
            fig, _ = plot_gantt_chart_matplotlib(instance, f"{name} Schedule")
            fig.savefig(f"{plots_prefix}_{name}_gantt.png")
            plt.close(fig)
    
    # Plot comparison
    if save_plots:
        fig, _ = plot_comparison_metrics(list(results.values()), "makespan")
        fig.savefig(f"{plots_prefix}_makespan_comparison.png")
        plt.close(fig)
        
        fig, _ = plot_comparison_metrics(list(results.values()), "total_flow_time")
        fig.savefig(f"{plots_prefix}_flowtime_comparison.png")
        plt.close(fig)
        
        # Plot GA convergence if available
        if "GA" in results and "best_fitness_history" in results["GA"]:
            fig, _ = plot_ga_convergence(results["GA"])
            fig.savefig(f"{plots_prefix}_ga_convergence.png")
            plt.close(fig)
    
    return results


def compare_problem_sizes(
    min_jobs: int = 3,
    max_jobs: int = 20,
    min_machines: int = 3,
    max_machines: int = 10,
    step: int = 3,
    runs_per_size: int = 3,
    seed: Optional[int] = None,
    save_plots: bool = False
) -> Dict[str, Any]:
    """
    Compare algorithm performance across different problem sizes.
    
    Args:
        min_jobs: Minimum number of jobs
        max_jobs: Maximum number of jobs
        min_machines: Minimum number of machines
        max_machines: Maximum number of machines
        step: Step size for jobs and machines
        runs_per_size: Number of runs per problem size
        seed: Random seed
        save_plots: Whether to save plots
        
    Returns:
        A dictionary with results from all experiments
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    results = {
        "problem_sizes": [],
        "FIFO": {"makespan": [], "time": []},
        "SPT": {"makespan": [], "time": []},
        "GA": {"makespan": [], "time": []}
    }
    
    # Create problem sizes
    job_sizes = list(range(min_jobs, max_jobs + 1, step))
    machine_sizes = list(range(min_machines, max_machines + 1, step))
    
    for n_jobs in job_sizes:
        for n_machines in machine_sizes:
            problem_size = f"{n_jobs}j_{n_machines}m"
            results["problem_sizes"].append(problem_size)
            
            print(f"\n{'='*50}")
            print(f"Testing problem size: {n_jobs} jobs, {n_machines} machines")
            print(f"{'='*50}")
            
            # Run multiple times and average
            size_results = {
                "FIFO": {"makespan": [], "time": []},
                "SPT": {"makespan": [], "time": []},
                "GA": {"makespan": [], "time": []}
            }
            
            for run in range(runs_per_size):
                # Generate a random instance
                instance = JSSPInstance.generate_random_instance(
                    n_jobs, n_machines, 
                    min_proc_time=1, 
                    max_proc_time=20,
                    seed=seed+run if seed is not None else None
                )
                
                # Run experiment
                exp_results = run_experiment(
                    instance, 
                    save_plots=(save_plots and run == 0),  # Save plots only for the first run
                    plots_prefix=f"size_{problem_size}_run{run}"
                )
                
                # Collect results
                for algo in ["FIFO", "SPT", "GA"]:
                    if algo in exp_results:
                        size_results[algo]["makespan"].append(exp_results[algo]["makespan"])
                        size_results[algo]["time"].append(exp_results[algo].get("computation_time", 0))
            
            # Calculate averages
            for algo in ["FIFO", "SPT", "GA"]:
                avg_makespan = sum(size_results[algo]["makespan"]) / len(size_results[algo]["makespan"])
                avg_time = sum(size_results[algo]["time"]) / len(size_results[algo]["time"])
                
                results[algo]["makespan"].append(avg_makespan)
                results[algo]["time"].append(avg_time)
                
                print(f"{algo} - Avg Makespan: {avg_makespan:.2f}, Avg Time: {avg_time:.4f}s")
    
    # Plot scaling trends if save_plots is True
    if save_plots:
        _plot_scaling_trends(results, "makespan", "Makespan Scaling with Problem Size")
        plt.savefig("scaling_makespan.png")
        plt.close()
        
        _plot_scaling_trends(results, "time", "Computation Time Scaling with Problem Size")
        plt.savefig("scaling_time.png")
        plt.close()
    
    return results


def _plot_scaling_trends(results: Dict[str, Any], metric: str, title: str):
    """Plot how algorithms scale with problem size."""
    plt.figure(figsize=(12, 6))
    
    problem_sizes = results["problem_sizes"]
    
    for algo in ["FIFO", "SPT", "GA"]:
        plt.plot(problem_sizes, results[algo][metric], marker='o', label=algo)
    
    plt.title(title)
    plt.xlabel("Problem Size (jobs_machines)")
    plt.ylabel(metric.title())
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()


if __name__ == "__main__":
    # Set random seed for reproducibility
    random_seed = 42
    
    print("=== JSSP Experiment Runner ===")
    print("\nRunning a simple experiment...")
    
    # Generate a small random instance
    small_instance = JSSPInstance.generate_random_instance(5, 3, seed=random_seed)
    run_experiment(small_instance, save_plots=True, plots_prefix="small_instance")
    
    print("\nComparing algorithm performance across different problem sizes...")
    results = compare_problem_sizes(
        min_jobs=3, 
        max_jobs=10, 
        min_machines=3, 
        max_machines=5, 
        step=3, 
        runs_per_size=2,
        seed=random_seed,
        save_plots=True
    )
    
    print("\nExperiments completed successfully.") 