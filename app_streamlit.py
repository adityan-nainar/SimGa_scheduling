"""
Streamlit application for Job Shop Scheduling Problem simulation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time
import random

from jssp.data import JSSPInstance, Job, Operation
from jssp.schedulers.simple import FIFOScheduler, SPTScheduler
from jssp.schedulers.genetic import GeneticScheduler
from jssp.visualization import (
    plot_comparison_metrics,
    plot_ga_convergence
)

# Set page configuration
st.set_page_config(
    page_title="JSSP Simulator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def extract_gantt_data(instance, algo_name):
    """Extract Gantt chart data from a scheduled instance."""
    data = []
    
    # Colors for different jobs
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
    ]
    
    for j, job in enumerate(instance.jobs):
        job_color = colors[j % len(colors)]
        
        for i, operation in enumerate(job.operations):
            if not operation.is_scheduled():
                continue
            
            data.append({
                'job': f'Job {job.job_id}',
                'machine': f'Machine {operation.machine_id}',
                'start': operation.start_time,
                'duration': operation.processing_time,
                'end': operation.end_time,
                'color': job_color,
                'text': f'J{job.job_id}-Op{i} ({operation.processing_time}t)'
            })
    
    return data

def run_simulation(problem_type, random_params, job_data):
    """Run the simulation with selected parameters."""
    # Create the problem instance
    if problem_type == "Random Problem":
        instance = JSSPInstance.generate_random_instance(
            random_params["num_jobs"],
            random_params["num_machines"],
            min_proc_time=random_params["min_proc_time"],
            max_proc_time=random_params["max_proc_time"],
            seed=random_params["seed"]
        )
    else:
        # Group by job ID
        jobs_by_id = {}
        for job_id, machine_id, proc_time in job_data:
            if job_id not in jobs_by_id:
                jobs_by_id[job_id] = []
            jobs_by_id[job_id].append((machine_id, proc_time))
        
        # Determine max machine ID
        max_machine_id = max(machine_id for _, machine_id, _ in job_data) + 1
        
        # Create instance
        instance = JSSPInstance(0, max_machine_id)
        
        # Create jobs
        for job_id in sorted(jobs_by_id.keys()):
            job = Job(job_id)
            for machine_id, proc_time in jobs_by_id[job_id]:
                job.add_operation(machine_id, proc_time)
            instance.add_job(job)
    
    # Run selected algorithms
    results = {}
    selected_algos = st.session_state.get("selected_algos", [])
    
    if "FIFO" in selected_algos:
        start_time = time.time()
        fifo_scheduler = FIFOScheduler()
        fifo_result = fifo_scheduler.schedule(instance.copy())
        fifo_result["computation_time"] = time.time() - start_time
        fifo_result["gantt_data"] = extract_gantt_data(instance, "FIFO")
        results["FIFO"] = fifo_result
    
    if "SPT" in selected_algos:
        start_time = time.time()
        spt_scheduler = SPTScheduler()
        instance_copy = instance.copy()
        spt_result = spt_scheduler.schedule(instance_copy)
        spt_result["computation_time"] = time.time() - start_time
        spt_result["gantt_data"] = extract_gantt_data(instance_copy, "SPT")
        results["SPT"] = spt_result
    
    if "Genetic Algorithm" in selected_algos:
        start_time = time.time()
        ga_scheduler = GeneticScheduler(
            population_size=st.session_state.get("population_size", 50),
            generations=st.session_state.get("generations", 100),
            crossover_prob=st.session_state.get("crossover_rate", 0.8),
            mutation_prob=st.session_state.get("mutation_rate", 0.2)
        )
        instance_copy = instance.copy()
        ga_result = ga_scheduler.schedule(instance_copy)
        ga_result["computation_time"] = time.time() - start_time
        ga_result["gantt_data"] = extract_gantt_data(instance_copy, "GA")
        results["Genetic Algorithm"] = ga_result
    
    return results

def display_results(results):
    """Display scheduling results."""
    if not results:
        st.warning("No results to display. Please run a simulation first.")
        return
    
    # Create results DataFrame
    results_data = []
    for algo_name, result in results.items():
        results_data.append({
            'Algorithm': algo_name,
            'Makespan': result["makespan"],
            'Total Flow Time': result["total_flow_time"],
            'Computation Time (s)': f"{result.get('computation_time', 0):.4f}"
        })
    
    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df, use_container_width=True)
    
    # Display schedules
    st.subheader("Schedules")
    tabs = st.tabs([algo_name for algo_name in results.keys()])
    
    for i, (tab, (algo_name, result)) in enumerate(zip(tabs, results.items())):
        with tab:
            st.write(f"Valid Schedule: {result['is_valid']}")
            
            # Display the Gantt chart
            df_data = result.get("gantt_data", [])
            
            if df_data:
                fig = go.Figure()
                
                for item in df_data:
                    fig.add_trace(go.Bar(
                        x=[item['duration']],
                        y=[item['machine']],
                        orientation='h',
                        base=item['start'],
                        marker=dict(color=item['color']),
                        text=item['text'],
                        name=item['job'],
                        showlegend=True,
                    ))
                
                fig.update_layout(
                    title=f"{algo_name} Schedule",
                    xaxis_title="Time",
                    yaxis_title="Machine",
                    barmode='overlay',
                    height=400,
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Display comparison charts if multiple algorithms
    if len(results) > 1:
        st.subheader("Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Makespan comparison
            fig, ax = plot_comparison_metrics(list(results.values()), "makespan")
            st.pyplot(fig)
        
        with col2:
            # Flow time comparison
            fig, ax = plot_comparison_metrics(list(results.values()), "total_flow_time")
            st.pyplot(fig)
        
        # Computation time comparison
        if "computation_time" in next(iter(results.values())):
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Extract data
            comp_time_values = [float(r.get("computation_time", 0)) for r in results.values()]
            comp_time_labels = list(results.keys())
            
            # Create bar chart
            bars = ax.bar(comp_time_labels, comp_time_values)
            
            # Add labels
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
    
    # Display GA convergence
    if "Genetic Algorithm" in results and "best_fitness_history" in results["Genetic Algorithm"]:
        st.subheader("Genetic Algorithm Convergence")
        fig, ax = plot_ga_convergence(results["Genetic Algorithm"])
        st.pyplot(fig)

def show_info():
    """Display information about JSSP."""
    st.markdown("""
    ## Job Shop Scheduling Problem (JSSP)
    
    The Job Shop Scheduling Problem is a classic optimization problem where:
    
    * A set of jobs must be processed on a set of machines
    * Each job consists of a sequence of operations
    * Each operation must be processed on a specific machine for a specific duration
    * Operations within a job must be processed in order
    * A machine can process only one operation at a time
    
    The goal is to find a schedule that minimizes objectives like:
    * **Makespan**: The total time to complete all jobs
    * **Flow Time**: The sum of completion times of all jobs
    
    ### Algorithms
    
    * **FIFO (First-In-First-Out)**: Schedules operations in the order they appear
    * **SPT (Shortest Processing Time)**: Prioritizes operations with shorter processing times
    * **Genetic Algorithm**: Uses evolutionary computation to search for near-optimal solutions
    
    ### Genetic Algorithm Details
    
    * **Representation**: Permutation of operations that respects job precedence
    * **Crossover**: Precedence-preserving crossover
    * **Mutation**: Swap mutation that maintains validity
    * **Selection**: Tournament selection
    * **Fitness**: Makespan (lower is better)
    """)

def main():
    st.title("Job Shop Scheduling Problem (JSSP) Simulator")
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Simulator", "Results", "About"])
    
    with tab1:
        st.header("Problem Configuration")
        
        # Sidebar for configuration
        with st.expander("Problem Configuration", expanded=True):
            # Problem type selection
            problem_type = st.radio(
                "Problem Type",
                ["Random Problem", "Custom Problem"],
                horizontal=True
            )
            
            if problem_type == "Random Problem":
                # Random problem parameters
                col1, col2 = st.columns(2)
                with col1:
                    num_jobs = st.slider("Number of Jobs", 2, 20, 5)
                    num_machines = st.slider("Number of Machines", 2, 10, 3)
                
                with col2:
                    min_proc_time = st.slider("Min Processing Time", 1, 20, 1)
                    max_proc_time = st.slider("Max Processing Time", 5, 50, 20)
                
                seed = st.number_input("Random Seed", 0, 999, 42)
                random_params = {
                    "num_jobs": num_jobs,
                    "num_machines": num_machines,
                    "min_proc_time": min_proc_time,
                    "max_proc_time": max_proc_time,
                    "seed": seed
                }
                job_data = None
            else:
                # Custom problem editor
                st.subheader("Job Operations Editor")
                
                # Initialize session state for job data if not exists
                if "job_data" not in st.session_state:
                    st.session_state.job_data = pd.DataFrame([
                        {"Job ID": 1, "Machine ID": 0, "Processing Time": 5},
                        {"Job ID": 1, "Machine ID": 1, "Processing Time": 10},
                        {"Job ID": 2, "Machine ID": 1, "Processing Time": 8},
                        {"Job ID": 2, "Machine ID": 0, "Processing Time": 6}
                    ])
                
                # Edit the dataframe
                edited_df = st.data_editor(
                    st.session_state.job_data,
                    num_rows="dynamic",
                    use_container_width=True
                )
                
                # Update session state
                st.session_state.job_data = edited_df
                
                # Convert dataframe to list
                job_data = []
                for _, row in edited_df.iterrows():
                    job_data.append([
                        int(row["Job ID"]),
                        int(row["Machine ID"]),
                        int(row["Processing Time"])
                    ])
                random_params = None
        
        with st.expander("Algorithm Configuration", expanded=True):
            # Algorithm selection
            st.subheader("Select Algorithms")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                fifo_selected = st.checkbox("FIFO", value=True)
            with col2:
                spt_selected = st.checkbox("SPT", value=True)
            with col3:
                ga_selected = st.checkbox("Genetic Algorithm", value=True)
            
            # Store selected algorithms in session state
            selected_algos = []
            if fifo_selected:
                selected_algos.append("FIFO")
            if spt_selected:
                selected_algos.append("SPT")
            if ga_selected:
                selected_algos.append("Genetic Algorithm")
            st.session_state.selected_algos = selected_algos
            
            # Genetic Algorithm parameters
            if ga_selected:
                st.subheader("Genetic Algorithm Parameters")
                
                col1, col2 = st.columns(2)
                with col1:
                    population_size = st.slider("Population Size", 10, 200, 50)
                    generations = st.slider("Generations", 10, 500, 100)
                
                with col2:
                    crossover_rate = st.slider("Crossover Rate", 0.1, 1.0, 0.8, 0.1)
                    mutation_rate = st.slider("Mutation Rate", 0.0, 1.0, 0.2, 0.1)
                
                # Store GA parameters in session state
                st.session_state.population_size = population_size
                st.session_state.generations = generations
                st.session_state.crossover_rate = crossover_rate
                st.session_state.mutation_rate = mutation_rate
        
        # Run simulation button
        if st.button("Run Simulation", type="primary", use_container_width=True):
            with st.spinner("Running simulation..."):
                results = run_simulation(problem_type, random_params, job_data)
                st.session_state.results = results
            st.success("Simulation completed!")
            st.rerun()
    
    with tab2:
        st.header("Results")
        # Display results if available
        if "results" in st.session_state:
            display_results(st.session_state.results)
        else:
            st.info("No results to display. Run a simulation in the Simulator tab first.")
    
    with tab3:
        st.header("About")
        show_info()

if __name__ == "__main__":
    main() 