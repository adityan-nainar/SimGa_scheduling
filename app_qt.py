"""
PyQt5 application for Job Shop Scheduling Problem simulation.
"""

import sys
import random
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import plotly.graph_objects as go
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QSpinBox, QCheckBox, QPushButton, QRadioButton, QGroupBox,
    QTableWidget, QTableWidgetItem, QSplitter, QComboBox, QScrollArea, QTextEdit,
    QSizePolicy, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWebEngineWidgets import QWebEngineView

from jssp.data import JSSPInstance, Job, Operation
from jssp.schedulers.simple import FIFOScheduler, SPTScheduler
from jssp.schedulers.genetic import GeneticScheduler
from jssp.visualization import (
    plot_comparison_metrics,
    plot_ga_convergence
)

class MatplotlibCanvas(FigureCanvas):
    """Canvas for embedding matplotlib plots in Qt."""
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(
            QSizePolicy.Expanding, 
            QSizePolicy.Expanding
        )
        self.updateGeometry()
        
        # Set minimum size to ensure the plot is visible
        self.setMinimumHeight(300)
        
    def sizeHint(self):
        w, h = self.get_width_height()
        return QSize(w, h)
        
    def minimumSizeHint(self):
        return QSize(400, 300)

class PlotlyWidget(QWebEngineView):
    """Widget for embedding Plotly plots in Qt."""
    def __init__(self, parent=None):
        super(PlotlyWidget, self).__init__(parent)
        self.setSizePolicy(
            QSizePolicy.Expanding, 
            QSizePolicy.Expanding
        )

    def plot(self, figure):
        html = figure.to_html(include_plotlyjs='cdn')
        self.setHtml(html)

class CustomTableWidget(QTableWidget):
    """Custom table widget with resizable behavior."""
    def __init__(self, parent=None):
        super(CustomTableWidget, self).__init__(parent)
        self.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )
        self.setMinimumHeight(200)

class JobEditorWidget(QWidget):
    """Widget for editing job operations manually."""
    def __init__(self, parent=None):
        super(JobEditorWidget, self).__init__(parent)
        
        layout = QVBoxLayout()
        
        # Create job operations table
        self.jobs_table = CustomTableWidget()
        self.jobs_table.setColumnCount(3)
        self.jobs_table.setHorizontalHeaderLabels(['Job ID', 'Machine ID', 'Processing Time'])
        
        # Initial data
        self.set_default_data()
        
        # Button to add a new row
        add_button = QPushButton("Add Operation")
        add_button.clicked.connect(self.add_row)
        
        layout.addWidget(QLabel("Define Operations:"))
        layout.addWidget(self.jobs_table)
        layout.addWidget(add_button)
        
        self.setLayout(layout)
    
    def set_default_data(self):
        default_data = [
            [1, 0, 5],
            [1, 1, 10],
            [2, 1, 8],
            [2, 0, 6]
        ]
        
        self.jobs_table.setRowCount(len(default_data))
        
        for row, (job_id, machine_id, proc_time) in enumerate(default_data):
            self.jobs_table.setItem(row, 0, QTableWidgetItem(str(job_id)))
            self.jobs_table.setItem(row, 1, QTableWidgetItem(str(machine_id)))
            self.jobs_table.setItem(row, 2, QTableWidgetItem(str(proc_time)))
    
    def add_row(self):
        row_count = self.jobs_table.rowCount()
        self.jobs_table.insertRow(row_count)
        
        # Populate with default values
        if row_count > 0:
            prev_job_id = int(self.jobs_table.item(row_count-1, 0).text())
            self.jobs_table.setItem(row_count, 0, QTableWidgetItem(str(prev_job_id)))
            self.jobs_table.setItem(row_count, 1, QTableWidgetItem("0"))
            self.jobs_table.setItem(row_count, 2, QTableWidgetItem("5"))
    
    def get_job_data(self):
        """Extract job data from the table."""
        data = []
        for row in range(self.jobs_table.rowCount()):
            try:
                job_id = int(self.jobs_table.item(row, 0).text())
                machine_id = int(self.jobs_table.item(row, 1).text())
                proc_time = int(self.jobs_table.item(row, 2).text())
                data.append([job_id, machine_id, proc_time])
            except (ValueError, AttributeError):
                # Skip invalid rows
                pass
        return data

class ResultsWidget(QWidget):
    """Widget to display scheduling results."""
    def __init__(self, parent=None):
        super(ResultsWidget, self).__init__(parent)
        
        layout = QVBoxLayout()
        
        # Results table
        self.results_table = CustomTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels([
            'Algorithm', 'Makespan', 'Total Flow Time', 'Computation Time (s)'
        ])
        
        # Tab widget for visualizations
        self.viz_tabs = QTabWidget()
        
        # Schedules tab
        self.schedules_tab = QWidget()
        self.schedules_layout = QVBoxLayout()
        self.schedules_tab.setLayout(self.schedules_layout)
        
        # Comparison tab
        self.comparison_tab = QScrollArea()
        self.comparison_tab.setWidgetResizable(True)
        self.comparison_content = QWidget()
        self.comparison_layout = QVBoxLayout(self.comparison_content)
        self.comparison_tab.setWidget(self.comparison_content)
        
        # GA Convergence tab
        self.convergence_tab = QWidget()
        self.convergence_layout = QVBoxLayout()
        self.convergence_tab.setLayout(self.convergence_layout)
        
        # Add tabs
        self.viz_tabs.addTab(self.schedules_tab, "Schedules")
        self.viz_tabs.addTab(self.comparison_tab, "Comparison")
        self.viz_tabs.addTab(self.convergence_tab, "GA Convergence")
        
        # Add widgets to layout
        layout.addWidget(QLabel("<h3>Results</h3>"))
        layout.addWidget(QLabel("Summary Metrics:"))
        layout.addWidget(self.results_table)
        layout.addWidget(self.viz_tabs)
        
        self.setLayout(layout)
    
    def clear_results(self):
        """Clear all results."""
        self.results_table.setRowCount(0)
        
        # Clear schedule tabs
        for i in reversed(range(self.schedules_layout.count())): 
            self.schedules_layout.itemAt(i).widget().setParent(None)
            
        # Clear comparison tab
        for i in reversed(range(self.comparison_layout.count())): 
            self.comparison_layout.itemAt(i).widget().setParent(None)
            
        # Clear convergence tab
        for i in reversed(range(self.convergence_layout.count())): 
            self.convergence_layout.itemAt(i).widget().setParent(None)
    
    def update_results(self, results):
        """Update results display with new data."""
        self.clear_results()
        
        # Update results table
        algorithms = list(results.keys())
        self.results_table.setRowCount(len(algorithms))
        
        for row, algo_name in enumerate(algorithms):
            result = results[algo_name]
            self.results_table.setItem(row, 0, QTableWidgetItem(algo_name))
            self.results_table.setItem(row, 1, QTableWidgetItem(str(result["makespan"])))
            self.results_table.setItem(row, 2, QTableWidgetItem(str(result["total_flow_time"])))
            self.results_table.setItem(row, 3, QTableWidgetItem(f"{result.get('computation_time', 0):.4f}"))
        
        # Update schedule tab
        for algo_name, result in results.items():
            # Add algorithm header
            schedule_label = QLabel(f"<h3>{algo_name} Schedule</h3>")
            self.schedules_layout.addWidget(schedule_label)
            
            # Create a Plotly widget for the Gantt chart
            gantt_plot = PlotlyWidget()
            
            # Create label to show if schedule is valid
            valid_label = QLabel(f"Valid Schedule: {result['is_valid']}")
            valid_label.setStyleSheet(
                "color: green;" if result['is_valid'] else "color: red;"
            )
            
            self.schedules_layout.addWidget(valid_label)
            self.schedules_layout.addWidget(gantt_plot)
            
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
                    width=800,
                )
                
                gantt_plot.plot(fig)
        
        # Update comparison tab
        if len(results) > 1:
            # Add makespan comparison
            makespan_label = QLabel("<h3>Makespan Comparison</h3>")
            self.comparison_layout.addWidget(makespan_label)
            
            makespan_canvas = MatplotlibCanvas(width=8, height=4)
            fig, ax = plot_comparison_metrics(list(results.values()), "makespan")
            makespan_canvas.figure = fig
            makespan_canvas.draw()
            self.comparison_layout.addWidget(makespan_canvas)
            
            # Add flow time comparison
            flowtime_label = QLabel("<h3>Flow Time Comparison</h3>")
            self.comparison_layout.addWidget(flowtime_label)
            
            flowtime_canvas = MatplotlibCanvas(width=8, height=4)
            fig, ax = plot_comparison_metrics(list(results.values()), "total_flow_time")
            flowtime_canvas.figure = fig
            flowtime_canvas.draw()
            self.comparison_layout.addWidget(flowtime_canvas)
            
            # Add computation time comparison if available
            if "computation_time" in next(iter(results.values())):
                comp_time_label = QLabel("<h3>Computation Time Comparison</h3>")
                self.comparison_layout.addWidget(comp_time_label)
                
                comp_time_canvas = MatplotlibCanvas(width=8, height=4)
                ax = comp_time_canvas.axes
                
                # Extract data
                comp_time_values = [r.get("computation_time", 0) for r in results.values()]
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
                
                comp_time_canvas.draw()
                self.comparison_layout.addWidget(comp_time_canvas)
        
        # Update GA convergence tab
        if "Genetic Algorithm" in results and "best_fitness_history" in results["Genetic Algorithm"]:
            ga_label = QLabel("<h3>Genetic Algorithm Convergence</h3>")
            self.convergence_layout.addWidget(ga_label)
            
            ga_canvas = MatplotlibCanvas(width=8, height=4)
            fig, ax = plot_ga_convergence(results["Genetic Algorithm"])
            ga_canvas.figure = fig
            ga_canvas.draw()
            self.convergence_layout.addWidget(ga_canvas)
        else:
            ga_label = QLabel("Genetic Algorithm was not run or convergence data is not available.")
            self.convergence_layout.addWidget(ga_label)

class InfoWidget(QWidget):
    """Widget to display information about JSSP."""
    def __init__(self, parent=None):
        super(InfoWidget, self).__init__(parent)
        
        layout = QVBoxLayout()
        
        text = """
        <h2>Job Shop Scheduling Problem (JSSP)</h2>
        
        <p>The Job Shop Scheduling Problem is a classic optimization problem where:</p>
        
        <ul>
            <li>A set of jobs must be processed on a set of machines</li>
            <li>Each job consists of a sequence of operations</li>
            <li>Each operation must be processed on a specific machine for a specific duration</li>
            <li>Operations within a job must be processed in order</li>
            <li>A machine can process only one operation at a time</li>
        </ul>
        
        <p>The goal is to find a schedule that minimizes objectives like:</p>
        <ul>
            <li><strong>Makespan</strong>: The total time to complete all jobs</li>
            <li><strong>Flow Time</strong>: The sum of completion times of all jobs</li>
        </ul>
        
        <h3>Algorithms</h3>
        
        <ul>
            <li><strong>FIFO (First-In-First-Out)</strong>: Schedules operations in the order they appear</li>
            <li><strong>SPT (Shortest Processing Time)</strong>: Prioritizes operations with shorter processing times</li>
            <li><strong>Genetic Algorithm</strong>: Uses evolutionary computation to search for near-optimal solutions</li>
        </ul>
        
        <h3>Genetic Algorithm Details</h3>
        
        <ul>
            <li><strong>Representation</strong>: Permutation of operations that respects job precedence</li>
            <li><strong>Crossover</strong>: Precedence-preserving crossover</li>
            <li><strong>Mutation</strong>: Swap mutation that maintains validity</li>
            <li><strong>Selection</strong>: Tournament selection</li>
            <li><strong>Fitness</strong>: Makespan (lower is better)</li>
        </ul>
        """
        
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setHtml(text)
        
        layout.addWidget(info_text)
        self.setLayout(layout)

class MainWindow(QMainWindow):
    """Main window for the JSSP application."""
    def __init__(self):
        super(MainWindow, self).__init__()
        
        self.setWindowTitle("JSSP Simulator")
        self.setMinimumSize(1200, 800)
        
        # Main layout with splitter
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel for configuration
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)
        left_panel.setMaximumWidth(400)
        
        # Problem configuration
        self.config_group = QGroupBox("Problem Configuration")
        config_layout = QVBoxLayout()
        
        # Problem type selection
        self.random_problem_radio = QRadioButton("Random Problem")
        self.custom_problem_radio = QRadioButton("Custom Problem")
        self.random_problem_radio.setChecked(True)
        
        config_layout.addWidget(self.random_problem_radio)
        config_layout.addWidget(self.custom_problem_radio)
        
        # Random problem parameters
        self.random_params_widget = QWidget()
        random_params_layout = QVBoxLayout()
        self.random_params_widget.setLayout(random_params_layout)
        
        # Number of jobs
        jobs_layout = QHBoxLayout()
        jobs_layout.addWidget(QLabel("Number of Jobs:"))
        self.jobs_spinbox = QSpinBox()
        self.jobs_spinbox.setRange(2, 20)
        self.jobs_spinbox.setValue(5)
        jobs_layout.addWidget(self.jobs_spinbox)
        random_params_layout.addLayout(jobs_layout)
        
        # Number of machines
        machines_layout = QHBoxLayout()
        machines_layout.addWidget(QLabel("Number of Machines:"))
        self.machines_spinbox = QSpinBox()
        self.machines_spinbox.setRange(2, 10)
        self.machines_spinbox.setValue(3)
        machines_layout.addWidget(self.machines_spinbox)
        random_params_layout.addLayout(machines_layout)
        
        # Min processing time
        min_proc_layout = QHBoxLayout()
        min_proc_layout.addWidget(QLabel("Min Processing Time:"))
        self.min_proc_spinbox = QSpinBox()
        self.min_proc_spinbox.setRange(1, 20)
        self.min_proc_spinbox.setValue(1)
        min_proc_layout.addWidget(self.min_proc_spinbox)
        random_params_layout.addLayout(min_proc_layout)
        
        # Max processing time
        max_proc_layout = QHBoxLayout()
        max_proc_layout.addWidget(QLabel("Max Processing Time:"))
        self.max_proc_spinbox = QSpinBox()
        self.max_proc_spinbox.setRange(5, 50)
        self.max_proc_spinbox.setValue(20)
        max_proc_layout.addWidget(self.max_proc_spinbox)
        random_params_layout.addLayout(max_proc_layout)
        
        # Random seed
        seed_layout = QHBoxLayout()
        seed_layout.addWidget(QLabel("Random Seed:"))
        self.seed_spinbox = QSpinBox()
        self.seed_spinbox.setRange(0, 999)
        self.seed_spinbox.setValue(42)
        seed_layout.addWidget(self.seed_spinbox)
        random_params_layout.addLayout(seed_layout)
        
        config_layout.addWidget(self.random_params_widget)
        
        # Custom problem editor
        self.custom_params_widget = JobEditorWidget()
        self.custom_params_widget.setVisible(False)
        config_layout.addWidget(self.custom_params_widget)
        
        self.config_group.setLayout(config_layout)
        
        # Algorithm configuration
        self.algo_group = QGroupBox("Algorithm Configuration")
        algo_layout = QVBoxLayout()
        
        # FIFO checkbox
        self.fifo_checkbox = QCheckBox("Use FIFO")
        self.fifo_checkbox.setChecked(True)
        algo_layout.addWidget(self.fifo_checkbox)
        
        # SPT checkbox
        self.spt_checkbox = QCheckBox("Use SPT")
        self.spt_checkbox.setChecked(True)
        algo_layout.addWidget(self.spt_checkbox)
        
        # GA checkbox and parameters
        self.ga_checkbox = QCheckBox("Use Genetic Algorithm")
        self.ga_checkbox.setChecked(True)
        algo_layout.addWidget(self.ga_checkbox)
        
        self.ga_params_widget = QWidget()
        ga_params_layout = QVBoxLayout()
        self.ga_params_widget.setLayout(ga_params_layout)
        
        # Population size
        pop_layout = QHBoxLayout()
        pop_layout.addWidget(QLabel("Population Size:"))
        self.population_spinbox = QSpinBox()
        self.population_spinbox.setRange(10, 200)
        self.population_spinbox.setValue(50)
        pop_layout.addWidget(self.population_spinbox)
        ga_params_layout.addLayout(pop_layout)
        
        # Generations
        gen_layout = QHBoxLayout()
        gen_layout.addWidget(QLabel("Generations:"))
        self.generations_spinbox = QSpinBox()
        self.generations_spinbox.setRange(10, 500)
        self.generations_spinbox.setValue(100)
        gen_layout.addWidget(self.generations_spinbox)
        ga_params_layout.addLayout(gen_layout)
        
        # Crossover rate
        crossover_layout = QHBoxLayout()
        crossover_layout.addWidget(QLabel("Crossover Rate:"))
        self.crossover_spinbox = QDoubleSpinBox()
        self.crossover_spinbox.setRange(0.1, 1.0)
        self.crossover_spinbox.setValue(0.8)
        self.crossover_spinbox.setSingleStep(0.1)
        crossover_layout.addWidget(self.crossover_spinbox)
        ga_params_layout.addLayout(crossover_layout)
        
        # Mutation rate
        mutation_layout = QHBoxLayout()
        mutation_layout.addWidget(QLabel("Mutation Rate:"))
        self.mutation_spinbox = QDoubleSpinBox()
        self.mutation_spinbox.setRange(0.0, 1.0)
        self.mutation_spinbox.setValue(0.2)
        self.mutation_spinbox.setSingleStep(0.1)
        mutation_layout.addWidget(self.mutation_spinbox)
        ga_params_layout.addLayout(mutation_layout)
        
        algo_layout.addWidget(self.ga_params_widget)
        
        self.algo_group.setLayout(algo_layout)
        
        # Run button
        self.run_button = QPushButton("Run Simulation")
        self.run_button.setStyleSheet("font-size: 14px; padding: 8px;")
        
        # Add widgets to left panel
        left_layout.addWidget(self.config_group)
        left_layout.addWidget(self.algo_group)
        left_layout.addWidget(self.run_button)
        left_layout.addStretch()
        
        # Right panel for results
        right_panel = QTabWidget()
        
        # Results tab
        self.results_widget = ResultsWidget()
        right_panel.addTab(self.results_widget, "Results")
        
        # Info tab
        self.info_widget = InfoWidget()
        right_panel.addTab(self.info_widget, "About")
        
        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        
        # Set initial splitter sizes
        splitter.setSizes([300, 700])
        
        main_layout.addWidget(splitter)
        
        # Connect signals
        self.connect_signals()
    
    def connect_signals(self):
        """Connect UI signals to slots."""
        self.random_problem_radio.toggled.connect(self.update_problem_type)
        self.custom_problem_radio.toggled.connect(self.update_problem_type)
        self.ga_checkbox.toggled.connect(self.ga_params_widget.setVisible)
        self.run_button.clicked.connect(self.run_simulation)
    
    def update_problem_type(self):
        """Update UI based on problem type selection."""
        is_random = self.random_problem_radio.isChecked()
        self.random_params_widget.setVisible(is_random)
        self.custom_params_widget.setVisible(not is_random)
    
    def run_simulation(self):
        """Run the simulation with selected parameters."""
        self.run_button.setEnabled(False)
        self.run_button.setText("Running...")
        
        # Create the problem instance
        if self.random_problem_radio.isChecked():
            instance = JSSPInstance.generate_random_instance(
                self.jobs_spinbox.value(),
                self.machines_spinbox.value(),
                min_proc_time=self.min_proc_spinbox.value(),
                max_proc_time=self.max_proc_spinbox.value(),
                seed=self.seed_spinbox.value()
            )
        else:
            # Create custom instance from table data
            job_data = self.custom_params_widget.get_job_data()
            
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
        
        if self.fifo_checkbox.isChecked():
            fifo_scheduler = FIFOScheduler()
            fifo_result = fifo_scheduler.schedule(instance.copy())
            fifo_result["gantt_data"] = self.extract_gantt_data(instance, "FIFO")
            results["FIFO"] = fifo_result
        
        if self.spt_checkbox.isChecked():
            spt_scheduler = SPTScheduler()
            instance_copy = instance.copy()
            spt_result = spt_scheduler.schedule(instance_copy)
            spt_result["gantt_data"] = self.extract_gantt_data(instance_copy, "SPT")
            results["SPT"] = spt_result
        
        if self.ga_checkbox.isChecked():
            ga_scheduler = GeneticScheduler(
                population_size=self.population_spinbox.value(),
                generations=self.generations_spinbox.value(),
                crossover_prob=self.crossover_spinbox.value(),
                mutation_prob=self.mutation_spinbox.value()
            )
            instance_copy = instance.copy()
            ga_result = ga_scheduler.schedule(instance_copy)
            ga_result["gantt_data"] = self.extract_gantt_data(instance_copy, "GA")
            results["Genetic Algorithm"] = ga_result
        
        # Update results
        self.results_widget.update_results(results)
        
        self.run_button.setText("Run Simulation")
        self.run_button.setEnabled(True)
    
    def extract_gantt_data(self, instance, algo_name):
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

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 