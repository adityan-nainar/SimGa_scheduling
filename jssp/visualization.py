"""
Visualization utilities for Job Shop Scheduling Problems.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
import pandas as pd
from typing import Dict, List, Any

from jssp.data import JSSPInstance


def plot_gantt_chart_matplotlib(instance: JSSPInstance, title: str = "Schedule Gantt Chart"):
    """
    Plot a Gantt chart of the schedule using Matplotlib.
    
    Args:
        instance: A scheduled JSSP instance
        title: Title of the chart
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Colors for different jobs
    cmap = plt.cm.get_cmap('tab20', len(instance.jobs))
    
    # Draw a rectangle for each scheduled operation
    for j, job in enumerate(instance.jobs):
        job_color = cmap(j)
        
        for i, operation in enumerate(job.operations):
            if not operation.is_scheduled():
                continue
                
            machine_id = operation.machine_id
            start_time = operation.start_time
            duration = operation.processing_time
            
            # Create a rectangle for this operation
            rect = patches.Rectangle(
                (start_time, machine_id - 0.4),
                duration,
                0.8,
                linewidth=1,
                edgecolor='black',
                facecolor=job_color,
                alpha=0.7
            )
            
            # Add the rectangle to the plot
            ax.add_patch(rect)
            
            # Add operation label
            ax.text(
                start_time + duration / 2,
                machine_id,
                f"J{job.job_id}",
                ha='center',
                va='center',
                color='black',
                fontweight='bold'
            )
    
    # Set the limits of the chart
    makespan = instance.makespan()
    ax.set_xlim(0, makespan * 1.05)
    ax.set_ylim(-0.5, len(instance.machines) - 0.5)
    
    # Set the labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Machine')
    ax.set_yticks(range(len(instance.machines)))
    ax.set_yticklabels([f'Machine {m.machine_id}' for m in instance.machines])
    ax.set_title(title)
    
    # Add a grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add a legend for jobs
    legend_patches = [
        patches.Patch(color=cmap(j), label=f'Job {job.job_id}')
        for j, job in enumerate(instance.jobs)
    ]
    ax.legend(handles=legend_patches, loc='upper right')
    
    plt.tight_layout()
    return fig, ax


def plot_gantt_chart_plotly(instance: JSSPInstance, title: str = "Schedule Gantt Chart"):
    """
    Plot an interactive Gantt chart of the schedule using Plotly.
    
    Args:
        instance: A scheduled JSSP instance
        title: Title of the chart
        
    Returns:
        A Plotly figure
    """
    # Collect data for the Gantt chart
    df_data = []
    
    for j, job in enumerate(instance.jobs):
        for i, operation in enumerate(job.operations):
            if not operation.is_scheduled():
                continue
                
            machine_id = operation.machine_id
            start_time = operation.start_time
            end_time = operation.end_time
            
            df_data.append({
                'Task': f'Machine {machine_id}',
                'Start': start_time,
                'Finish': end_time,
                'Job': f'Job {job.job_id}',
                'Operation': f'Op {i}'
            })
    
    # Create a DataFrame
    df = pd.DataFrame(df_data)
    
    if df.empty:
        # If no operations are scheduled, create an empty chart
        fig = px.bar(title=title + " (No operations scheduled)")
        return fig
    
    # Create a Gantt chart
    fig = ff.create_gantt(
        df,
        colors=px.colors.qualitative.Plotly[:len(instance.jobs)],
        index_col='Job',
        show_colorbar=True,
        group_tasks=True,
        title=title
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Time',
        yaxis_title='',
        autosize=True,
        margin=dict(l=10, r=10, b=10, t=50),
        legend_title_text='Jobs'
    )
    
    return fig


def plot_comparison_metrics(results: List[Dict[str, Any]], metric: str = 'makespan'):
    """
    Plot a comparison of scheduling results.
    
    Args:
        results: List of scheduling results
        metric: Metric to compare ('makespan', 'total_flow_time', 'average_flow_time')
        
    Returns:
        A Matplotlib figure and axes
    """
    # Extract data for plotting
    algorithms = [r['algorithm'] for r in results]
    values = [r[metric] for r in results]
    
    # Create a bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(algorithms, values, color=plt.cm.tab10(range(len(results))))
    
    # Add values on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.1,
            f'{values[i]:.2f}',
            ha='center',
            va='bottom'
        )
    
    # Set labels and title
    ax.set_xlabel('Algorithm')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'Comparison of {metric.replace("_", " ").title()}')
    
    # Add a grid
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    plt.tight_layout()
    return fig, ax


def plot_ga_convergence(result: Dict[str, Any]):
    """
    Plot the convergence of the genetic algorithm.
    
    Args:
        result: Result from the GeneticScheduler
        
    Returns:
        A Matplotlib figure and axes
    """
    # Check if the result contains convergence data
    if 'best_fitness_history' not in result or 'avg_fitness_history' not in result:
        raise ValueError("The result does not contain convergence data")
    
    best_fitness = result['best_fitness_history']
    avg_fitness = result['avg_fitness_history']
    
    # Create a line chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(best_fitness, label='Best Fitness', color='green', linewidth=2)
    ax.plot(avg_fitness, label='Average Fitness', color='blue', linewidth=1, alpha=0.7)
    
    # Set labels and title
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness (Makespan)')
    ax.set_title('Genetic Algorithm Convergence')
    
    # Add a grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add a legend
    ax.legend()
    
    plt.tight_layout()
    return fig, ax 