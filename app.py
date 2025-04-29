"""
Streamlit app for Job Shop Scheduling Problem simulation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from jssp.data import JSSPInstance
from jssp.schedulers.simple import FIFOScheduler, SPTScheduler
from jssp.schedulers.genetic import GeneticScheduler
from jssp.visualization import (
    plot_gantt_chart_plotly,
    plot_comparison_metrics,
    plot_ga_convergence
)

# Set page config
st.set_page_config(
    page_title="JSSP Simulator",
    page_icon="🏭",
    layout="wide",
)

# Create sidebar
st.sidebar.title("JSSP Simulator")
st.sidebar.markdown("Optimize job scheduling with Genetic Algorithms")

# Problem generation options
st.sidebar.header("Problem Configuration")

problem_tab = st.sidebar.radio("Problem Type", ["Random Problem", "Custom Problem"])

if problem_tab == "Random Problem":
    num_jobs = st.sidebar.slider("Number of Jobs", 2, 20, 5)
    num_machines = st.sidebar.slider("Number of Machines", 2, 10, 3)
    min_proc_time = st.sidebar.slider("Min Processing Time", 1, 10, 1)
    max_proc_time = st.sidebar.slider("Max Processing Time", 5, 50, 20)
    random_seed = st.sidebar.number_input("Random Seed", 0, 999, 42)

# Algorithm options
st.sidebar.header("Algorithm Configuration")
use_fifo = st.sidebar.checkbox("Use FIFO", True)
use_spt = st.sidebar.checkbox("Use SPT", True)
use_ga = st.sidebar.checkbox("Use Genetic Algorithm", True)

if use_ga:
    ga_population = st.sidebar.slider("GA Population Size", 10, 200, 50)
    ga_generations = st.sidebar.slider("GA Generations", 10, 500, 100)
    ga_crossover = st.sidebar.slider("GA Crossover Rate", 0.1, 1.0, 0.8)
    ga_mutation = st.sidebar.slider("GA Mutation Rate", 0.0, 1.0, 0.2)

# Main interface
st.title("Job Shop Scheduling Problem Simulator")

if problem_tab == "Random Problem":
    st.markdown(f"""
    ### Problem Configuration
    - **Jobs**: {num_jobs}
    - **Machines**: {num_machines}
    - **Processing Time Range**: {min_proc_time} to {max_proc_time}
    """)
else:
    st.markdown("""
    ### Custom Problem Configuration
    Define your own problem structure below:
    """)
    # Create a template for custom job definition
    st.markdown("#### Job Definition")
    
    # Dynamic job/machine input using dataframes
    if 'custom_jobs' not in st.session_state:
        # Initialize with a small template
        st.session_state.custom_jobs = pd.DataFrame(
            [[1, 0, 5], [1, 1, 10], [2, 1, 8], [2, 0, 6]],
            columns=['Job ID', 'Machine ID', 'Processing Time']
        )
    
    # Edit the dataframe directly
    edited_df = st.data_editor(
        st.session_state.custom_jobs,
        num_rows="dynamic",
        use_container_width=True,
    )
    st.session_state.custom_jobs = edited_df

# Run button
run_col1, run_col2 = st.columns([1, 5])
with run_col1:
    run_button = st.button("Run Simulation", type="primary")

# Container for results
results_container = st.container()

if run_button:
    with st.spinner("Running simulation..."):
        # Create the problem instance
        if problem_tab == "Random Problem":
            instance = JSSPInstance.generate_random_instance(
                num_jobs, num_machines, 
                min_proc_time=min_proc_time, 
                max_proc_time=max_proc_time,
                seed=random_seed
            )
        else:
            # Create custom instance from dataframe
            df = st.session_state.custom_jobs
            
            # Group by job ID
            job_groups = df.groupby('Job ID')
            
            # Create instance with the maximum machine ID
            max_machine_id = int(df['Machine ID'].max()) + 1
            instance = JSSPInstance(0, max_machine_id)
            
            # Add jobs
            for job_id, group in job_groups:
                job = instance.jobs.append(instance.jobs[-1] if instance.jobs else None)
                if not job:
                    job = instance.add_job()
                
                for _, row in group.iterrows():
                    job.add_operation(int(row['Machine ID']), int(row['Processing Time']))
        
        # Run selected algorithms
        results = {}
        
        if use_fifo:
            fifo_scheduler = FIFOScheduler()
            fifo_result = fifo_scheduler.schedule(instance.copy() if hasattr(instance, 'copy') else instance)
            results["FIFO"] = fifo_result
        
        if use_spt:
            spt_scheduler = SPTScheduler()
            spt_result = spt_scheduler.schedule(instance.copy() if hasattr(instance, 'copy') else instance)
            results["SPT"] = spt_result
            
        if use_ga:
            ga_scheduler = GeneticScheduler(
                population_size=ga_population,
                generations=ga_generations,
                crossover_prob=ga_crossover,
                mutation_prob=ga_mutation
            )
            ga_result = ga_scheduler.schedule(instance.copy() if hasattr(instance, 'copy') else instance)
            results["Genetic Algorithm"] = ga_result
        
        # Display results
        with results_container:
            st.markdown("## Results")
            
            # Summary metrics
            st.markdown("### Summary Metrics")
            metrics_df = pd.DataFrame({
                "Algorithm": list(results.keys()),
                "Makespan": [r["makespan"] for r in results.values()],
                "Total Flow Time": [r["total_flow_time"] for r in results.values()],
                "Valid Schedule": [r["is_valid"] for r in results.values()],
                "Computation Time (s)": [r.get("computation_time", "N/A") for r in results.values()]
            })
            
            st.dataframe(metrics_df, use_container_width=True)
            
            # Create tabs for different visualizations
            viz_tabs = st.tabs(["Schedules", "Comparison", "GA Convergence"])
            
            # Schedules tab
            with viz_tabs[0]:
                for algo_name, result in results.items():
                    st.markdown(f"### {algo_name} Schedule")
                    
                    if algo_name == "FIFO":
                        scheduler = FIFOScheduler()
                        instance_copy = instance.copy() if hasattr(instance, 'copy') else instance
                        scheduler.schedule(instance_copy)
                        fig = plot_gantt_chart_plotly(instance_copy, f"{algo_name} Schedule")
                    elif algo_name == "SPT":
                        scheduler = SPTScheduler()
                        instance_copy = instance.copy() if hasattr(instance, 'copy') else instance
                        scheduler.schedule(instance_copy)
                        fig = plot_gantt_chart_plotly(instance_copy, f"{algo_name} Schedule")
                    else:  # GA
                        scheduler = GeneticScheduler(
                            population_size=ga_population,
                            generations=ga_generations,
                            crossover_prob=ga_crossover,
                            mutation_prob=ga_mutation
                        )
                        instance_copy = instance.copy() if hasattr(instance, 'copy') else instance
                        scheduler.schedule(instance_copy)
                        fig = plot_gantt_chart_plotly(instance_copy, f"{algo_name} Schedule")
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Comparison tab
            with viz_tabs[1]:
                st.markdown("### Algorithm Comparison")
                
                fig, ax = plot_comparison_metrics(list(results.values()), "makespan")
                st.pyplot(fig)
                
                fig, ax = plot_comparison_metrics(list(results.values()), "total_flow_time")
                st.pyplot(fig)
                
                if "computation_time" in results.get("FIFO", {}):
                    comp_time_values = [r.get("computation_time", 0) for r in results.values()]
                    comp_time_labels = list(results.keys())
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.bar(comp_time_labels, comp_time_values)
                    
                    for i, bar in enumerate(bars):
                        height = bar.get_height()
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            height + 0.001,
                            f'{comp_time_values[i]:.4f}',
                            ha='center',
                            va='bottom'
                        )
                    
                    ax.set_xlabel('Algorithm')
                    ax.set_ylabel('Computation Time (s)')
                    ax.set_title('Comparison of Computation Time')
                    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
                    
                    st.pyplot(fig)
            
            # GA Convergence tab
            with viz_tabs[2]:
                st.markdown("### Genetic Algorithm Convergence")
                
                if "Genetic Algorithm" in results and "best_fitness_history" in results["Genetic Algorithm"]:
                    fig, ax = plot_ga_convergence(results["Genetic Algorithm"])
                    st.pyplot(fig)
                else:
                    st.info("Genetic Algorithm was not run or convergence data is not available.")
else:
    with results_container:
        st.info("Click 'Run Simulation' to see results.")

# Add documentation
with st.expander("About Job Shop Scheduling"):
    st.markdown("""
    ## Job Shop Scheduling Problem (JSSP)
    
    The Job Shop Scheduling Problem is a classic optimization problem where:
    
    - A set of jobs must be processed on a set of machines
    - Each job consists of a sequence of operations
    - Each operation must be processed on a specific machine for a specific duration
    - Operations within a job must be processed in order
    - A machine can process only one operation at a time
    
    The goal is to find a schedule that minimizes objectives like:
    - **Makespan**: The total time to complete all jobs
    - **Flow Time**: The sum of completion times of all jobs
    
    ### Algorithms
    
    - **FIFO (First-In-First-Out)**: Schedules operations in the order they appear
    - **SPT (Shortest Processing Time)**: Prioritizes operations with shorter processing times
    - **Genetic Algorithm**: Uses evolutionary computation to search for near-optimal solutions
    
    ### Genetic Algorithm Details
    
    - **Representation**: Permutation of operations that respects job precedence
    - **Crossover**: Precedence-preserving crossover
    - **Mutation**: Swap mutation that maintains validity
    - **Selection**: Tournament selection
    - **Fitness**: Makespan (lower is better)
    """) 