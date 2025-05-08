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
import inspect

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

def run_simulation(problem_type, random_params, job_data, use_uncertainty=False, uncertainty_params=None):
    """Run the simulation with selected parameters."""
    # Create the problem instance
    if problem_type == "Random Problem":
        # Determine which uncertainty parameters to use
        arrival_time_method = None
        lambda_arrival = 0.1
        max_arrival_window = 100
        proc_time_variability = 0.0
        machine_failure_rate = 0.0
        min_repair_time = 5
        max_repair_time = 20
        simulation_horizon = 1000
        
        if use_uncertainty and uncertainty_params:
            if uncertainty_params.get("arrival_time_method") != "None":
                arrival_time_method = uncertainty_params.get("arrival_time_method").lower()
                lambda_arrival = uncertainty_params.get("lambda_arrival", 0.1)
                max_arrival_window = uncertainty_params.get("max_arrival_window", 100)
                
            proc_time_variability = uncertainty_params.get("proc_time_variability", 0.0)
            machine_failure_rate = uncertainty_params.get("machine_failure_rate", 0.0)
            min_repair_time = uncertainty_params.get("min_repair_time", 5)
            max_repair_time = uncertainty_params.get("max_repair_time", 20)
            simulation_horizon = random_params["max_proc_time"] * random_params["num_jobs"] * 5  # Estimate
        
        # Generate the instance with uncertainty parameters
        instance = JSSPInstance.generate_random_instance(
            random_params["num_jobs"],
            random_params["num_machines"],
            min_proc_time=random_params["min_proc_time"],
            max_proc_time=random_params["max_proc_time"],
            seed=random_params["seed"],
            arrival_time_method=arrival_time_method,
            lambda_arrival=lambda_arrival,
            max_arrival_window=max_arrival_window,
            proc_time_variability=proc_time_variability,
            machine_failure_rate=machine_failure_rate,
            min_repair_time=min_repair_time,
            max_repair_time=max_repair_time,
            simulation_horizon=simulation_horizon
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
            
        # Apply uncertainty if needed
        if use_uncertainty and uncertainty_params:
            # Set job arrival times
            if uncertainty_params.get("arrival_time_method") != "None":
                arrival_method = uncertainty_params.get("arrival_time_method").lower()
                lambda_arr = uncertainty_params.get("lambda_arrival", 0.1)
                max_window = uncertainty_params.get("max_arrival_window", 100)
                instance.generate_job_arrival_times(
                    method=arrival_method,
                    lambda_arrival=lambda_arr,
                    max_arrival_window=max_window
                )
                
            # Set processing time variability
            proc_variability = uncertainty_params.get("proc_time_variability", 0.0)
            if proc_variability > 0:
                instance.set_processing_time_variability(proc_variability)
                
            # Set machine breakdowns
            failure_rate = uncertainty_params.get("machine_failure_rate", 0.0)
            if failure_rate > 0:
                min_repair = uncertainty_params.get("min_repair_time", 5)
                max_repair = uncertainty_params.get("max_repair_time", 20)
                instance.setup_machine_breakdowns(failure_rate, min_repair, max_repair)
                
                # Estimate simulation horizon
                makespan_estimate = sum(job.total_processing_time() for job in instance.jobs)
                instance.generate_machine_breakdowns(makespan_estimate * 2)
    
    # Run selected algorithms
    results = {}
    selected_algos = st.session_state.get("selected_algos", [])
    
    # Get uncertainty simulation parameters
    num_simulations = 1
    stability_weight = 0.5
    if use_uncertainty and uncertainty_params:
        num_simulations = uncertainty_params.get("num_simulations", 5)
        stability_weight = uncertainty_params.get("stability_weight", 0.5)
    
    if "FIFO" in selected_algos:
        start_time = time.time()
        fifo_scheduler = FIFOScheduler(
            use_uncertainty=use_uncertainty,
            num_simulations=num_simulations,
            stability_weight=stability_weight
        )
        fifo_result = fifo_scheduler.schedule(instance.copy())
        fifo_result["computation_time"] = time.time() - start_time
        fifo_result["gantt_data"] = extract_gantt_data(instance, "FIFO")
        results["FIFO"] = fifo_result
    
    if "SPT" in selected_algos:
        start_time = time.time()
        spt_scheduler = SPTScheduler(
            use_uncertainty=use_uncertainty,
            num_simulations=num_simulations,
            stability_weight=stability_weight
        )
        instance_copy = instance.copy()
        spt_result = spt_scheduler.schedule(instance_copy)
        spt_result["computation_time"] = time.time() - start_time
        spt_result["gantt_data"] = extract_gantt_data(instance_copy, "SPT")
        results["SPT"] = spt_result
    
    if "Genetic Algorithm" in selected_algos:
        start_time = time.time()
        buffer_mutation_rate = st.session_state.get("buffer_mutation_rate", 0.1) if use_uncertainty else 0.0
        ga_scheduler = GeneticScheduler(
            population_size=st.session_state.get("population_size", 50),
            generations=st.session_state.get("generations", 100),
            crossover_prob=st.session_state.get("crossover_rate", 0.8),
            mutation_prob=st.session_state.get("mutation_rate", 0.2),
            buffer_mutation_prob=buffer_mutation_rate,
            use_uncertainty=use_uncertainty,
            num_simulations=num_simulations,
            stability_weight=stability_weight
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
            col3, col4 = st.columns(2)
            
            with col3:
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
            
            # Display GA convergence in the fourth quadrant of the grid
            with col4:
                if "Genetic Algorithm" in results and "best_fitness_history" in results["Genetic Algorithm"]:
                    fig, ax = plot_ga_convergence(results["Genetic Algorithm"])
                    st.pyplot(fig)
                else:
                    # Create a placeholder plot if GA results not available
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.text(0.5, 0.5, "GA Convergence data not available", 
                            ha='center', va='center', fontsize=12)
                    ax.set_title("Genetic Algorithm Convergence")
                    ax.set_xlabel("Generation")
                    ax.set_ylabel("Fitness")
                    ax.grid(False)
                    st.pyplot(fig)
    
    # Only display GA convergence separately if we don't have multiple algorithms for comparison
    elif "Genetic Algorithm" in results and "best_fitness_history" in results["Genetic Algorithm"]:
        st.subheader("Genetic Algorithm Convergence")
        fig, ax = plot_ga_convergence(results["Genetic Algorithm"])
        st.pyplot(fig)

def show_key_terms():
    """Display explanations of key JSSP terms."""
    st.markdown("""
    ## Key Terms in Job Shop Scheduling

    ### Performance Metrics
    - **Makespan**: The total completion time of all jobs (time when the last job finishes). This is the most common objective to minimize.
    - **Flow Time**: The time a job spends in the system from start to finish, including waiting time.
    - **Total Flow Time**: The sum of flow times for all jobs.
    - **Average Flow Time**: The average time each job spends in the system.
    - **Tardiness**: The amount of time a job is completed after its due date.
    - **Lateness**: The difference between completion time and due date (can be negative).
    - **Weighted Metrics**: Any of the above metrics where jobs have different priorities or weights.

    ### Scheduling Concepts
    - **Operation**: A single task that must be performed on a specific machine for a specific duration.
    - **Job**: A sequence of operations that must be processed in a specific order.
    - **Machine**: A resource that can process one operation at a time.
    - **Schedule**: An assignment of operations to machines with specific start times.
    - **Feasible Schedule**: A schedule that satisfies all constraints (no overlaps, correct sequence).
    - **Optimal Schedule**: A feasible schedule that minimizes the objective function.

    ### Scheduling Rules
    - **FIFO (First-In-First-Out)**: Operations are processed in the order they become available.
    - **SPT (Shortest Processing Time)**: Operations with shorter processing times are scheduled first.
    - **LPT (Longest Processing Time)**: Operations with longer processing times are scheduled first.
    - **EDD (Earliest Due Date)**: Jobs with earlier due dates are prioritized.

    ### Advanced Techniques
    - **Genetic Algorithm**: Evolutionary approach that generates, evolves, and selects schedules.
    - **Simulated Annealing**: Probabilistic technique that mimics the physical annealing process.
    - **Tabu Search**: Meta-heuristic that uses memory structures to avoid cycling in local searches.
    - **Branch and Bound**: Systematic method that explores the solution space by branching decisions.
    """)

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
    
    # Sidebar configuration
    st.sidebar.title("Problem Configuration")
    
    # Problem type selection
    problem_type = st.sidebar.radio(
        "Problem Type",
        ["Random Problem", "Custom Problem"],
        horizontal=False
    )
    
    if problem_type == "Random Problem":
        # Random problem parameters
        random_params = {}
        random_params["num_jobs"] = st.sidebar.slider("Number of Jobs", 2, 100, 5)
        random_params["num_machines"] = st.sidebar.slider("Number of Machines", 2, 50, 3)
        random_params["min_proc_time"] = st.sidebar.slider("Min Processing Time", 1, 30, 1)
        random_params["max_proc_time"] = st.sidebar.slider("Max Processing Time", 5, 80, 20)
        random_params["seed"] = st.sidebar.number_input("Random Seed", 0, 999, 42)
        job_data = None
    else:
        # Custom problem editor
        st.sidebar.subheader("Job Operations Editor")
        
        # Initialize session state for job data if not exists
        if "job_data" not in st.session_state:
            st.session_state.job_data = pd.DataFrame([
                {"Job ID": 1, "Machine ID": 0, "Processing Time": 5},
                {"Job ID": 1, "Machine ID": 1, "Processing Time": 10},
                {"Job ID": 2, "Machine ID": 1, "Processing Time": 8},
                {"Job ID": 2, "Machine ID": 0, "Processing Time": 6}
            ])
        
        # Edit the dataframe
        edited_df = st.sidebar.data_editor(
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
    
    # Algorithm selection
    st.sidebar.title("Algorithm Configuration")
    
    fifo_selected = st.sidebar.checkbox("FIFO", value=True)
    spt_selected = st.sidebar.checkbox("SPT", value=True)
    ga_selected = st.sidebar.checkbox("Genetic Algorithm", value=True)
    
    # Store selected algorithms in session state
    selected_algos = []
    if fifo_selected:
        selected_algos.append("FIFO")
    if spt_selected:
        selected_algos.append("SPT")
    if ga_selected:
        selected_algos.append("Genetic Algorithm")
    st.session_state.selected_algos = selected_algos
    
    # Uncertainty Parameters
    st.sidebar.title("Uncertainty Configuration")
    
    use_uncertainty = st.sidebar.checkbox("Enable Uncertainty", value=False)
    
    # Initialize uncertainty parameters
    uncertainty_params = {}
    
    if use_uncertainty:
        # Job Arrival Times
        st.sidebar.subheader("Job Arrival Times")
        arrival_time_method = st.sidebar.selectbox(
            "Job Arrival Distribution", 
            ["None", "Uniform", "Exponential"],
            index=0
        )
        
        uncertainty_params["arrival_time_method"] = arrival_time_method
        
        if arrival_time_method != "None":
            if arrival_time_method == "Exponential":
                lambda_arrival = st.sidebar.slider(
                    "λ (arrival rate)", 
                    0.01, 1.0, 0.1, 0.01,
                    help="Higher values = more frequent arrivals"
                )
                uncertainty_params["lambda_arrival"] = lambda_arrival
            else:  # Uniform
                max_arrival_window = st.sidebar.slider(
                    "Max Arrival Window", 
                    10, 500, 100, 10,
                    help="Maximum time for job arrivals"
                )
                uncertainty_params["max_arrival_window"] = max_arrival_window
        
        # Processing Time Variability
        st.sidebar.subheader("Processing Time Variability")
        proc_time_variability = st.sidebar.slider(
            "Processing Time Variability (±%)", 
            0, 100, 20, 5
        ) / 100.0  # Convert to fraction
        
        uncertainty_params["proc_time_variability"] = proc_time_variability
        
        # Machine Breakdowns
        st.sidebar.subheader("Machine Breakdowns")
        machine_failure_rate = st.sidebar.slider(
            "Machine Failure Rate", 
            0.0, 0.1, 0.01, 0.001,
            help="Average failures per time unit"
        )
        
        uncertainty_params["machine_failure_rate"] = machine_failure_rate
        
        if machine_failure_rate > 0:
            min_repair_time = st.sidebar.slider("Min Repair Time", 1, 50, 5)
            max_repair_time = st.sidebar.slider("Max Repair Time", min_repair_time, 100, min_repair_time + 15)
            
            uncertainty_params["min_repair_time"] = min_repair_time
            uncertainty_params["max_repair_time"] = max_repair_time
        
        # Simulation Parameters
        st.sidebar.subheader("Simulation Parameters")
        num_simulations = st.sidebar.slider(
            "Simulation Runs (K)", 
            1, 20, 5,
            help="Number of simulations for evaluation"
        )
        
        uncertainty_params["num_simulations"] = num_simulations
        
        stability_weight = st.sidebar.slider(
            "Stability Weight (α)", 
            0.0, 2.0, 0.5, 0.1,
            help="Weight of standard deviation in fitness"
        )
        
        uncertainty_params["stability_weight"] = stability_weight
    
    # Genetic Algorithm parameters
    if ga_selected:
        st.sidebar.subheader("Genetic Algorithm Parameters")
        
        st.session_state.population_size = st.sidebar.slider("Population Size", 10, 200, 50)
        st.session_state.generations = st.sidebar.slider("Generations", 10, 500, 100)
        st.session_state.crossover_rate = st.sidebar.slider("Crossover Rate", 0.1, 1.0, 0.8, 0.1)
        st.session_state.mutation_rate = st.sidebar.slider("Mutation Rate", 0.0, 1.0, 0.2, 0.1)
        
        if use_uncertainty:
            st.session_state.buffer_mutation_rate = st.sidebar.slider(
                "Buffer Mutation Rate", 
                0.0, 0.5, 0.1, 0.05,
                help="Probability of inserting delay buffers"
            )
            
            uncertainty_params["buffer_mutation_rate"] = st.session_state.buffer_mutation_rate
    
    # Run simulation button
    if st.sidebar.button("Run Simulation", type="primary", use_container_width=True):
        with st.spinner("Running simulation..."):
            results = run_simulation(
                problem_type, 
                random_params, 
                job_data, 
                use_uncertainty, 
                uncertainty_params
            )
            st.session_state.results = results
        st.sidebar.success("Simulation completed!")
    
    # Main content area - tabs for results, about, and key terms
    tab1, tab2, tab3, tab4 = st.tabs(["Results", "About JSSP", "Key Terms", "Uncertainty"])
    
    with tab1:
        # Display results if available
        if "results" in st.session_state:
            display_results(st.session_state.results)
        else:
            st.info("Configure parameters in the sidebar and click 'Run Simulation' to see results.")
    
    with tab2:
        show_info()
    
    with tab3:
        show_key_terms()
        
    with tab4:
        show_uncertainty_info()

if __name__ == "__main__":
    main() 