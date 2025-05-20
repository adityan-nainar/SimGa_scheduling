"""
Scheduling algorithms for Job Shop Scheduling Problems.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
import random
import statistics
import enum

from jssp.data import JSSPInstance

class EventType(enum.Enum):
    """Types of events in the event-driven simulator."""
    JOB_ARRIVAL = 1      # Job arrives to the system
    OPERATION_START = 2  # Operation starts processing
    OPERATION_END = 3    # Operation finishes processing
    MACHINE_DOWN = 4     # Machine breaks down
    MACHINE_UP = 5       # Machine is repaired and available again
    BUFFER_END = 6       # A buffer period ends

class Event:
    """Represents an event in the event-driven simulator."""
    
    def __init__(self, event_type: EventType, time: float, job_id: Optional[int] = None, 
                 machine_id: Optional[int] = None, operation_id: Optional[int] = None,
                 duration: Optional[float] = None):
        self.event_type = event_type
        self.time = time
        self.job_id = job_id
        self.machine_id = machine_id
        self.operation_id = operation_id
        self.duration = duration
        
    def __lt__(self, other: 'Event') -> bool:
        """Compare events by time for priority queue ordering."""
        return self.time < other.time

class Scheduler(ABC):
    """Base class for all scheduling algorithms."""
    
    def __init__(self, use_uncertainty: bool = False, num_simulations: int = 1, stability_weight: float = 0.5):
        self.use_uncertainty = use_uncertainty  # Whether to consider uncertainty
        self.num_simulations = num_simulations  # Number of simulations to run for evaluating schedules
        self.stability_weight = stability_weight  # Weight of standard deviation in fitness calculation (alpha)
    
    @abstractmethod
    def schedule(self, instance: JSSPInstance) -> Dict[str, Any]:
        """
        Schedule the jobs on machines.
        
        Args:
            instance: A JSSP instance with jobs and machines
            
        Returns:
            A dictionary with scheduling results and metrics
        """
        pass
    
    def evaluate_with_uncertainty(self, instance: JSSPInstance) -> Dict[str, Any]:
        """
        Evaluate a schedule with multiple simulations to account for uncertainty.
        
        Args:
            instance: A JSSP instance with a schedule
            
        Returns:
            Dictionary with average makespan, standard deviation, and other metrics
        """
        print("    Starting evaluate_with_uncertainty method")
        # If uncertainty is not enabled, just return the basic metrics
        if not self.use_uncertainty:
            print("    Uncertainty disabled, returning basic metrics")
            makespan = instance.makespan()
            total_flow_time = instance.total_flow_time()
            return {
                "makespan": makespan,
                "makespan_std": 0,
                "total_flow_time": total_flow_time,
                "flow_time_std": 0,
                "robust_fitness": makespan,  # No penalty when uncertainty is disabled
                "makespans": [makespan],
                "flow_times": [total_flow_time]
            }
        
        print(f"    Running {self.num_simulations} uncertainty simulations")
        makespans = []
        flow_times = []
        
        for i in range(self.num_simulations):
            print(f"    Uncertainty simulation {i+1}/{self.num_simulations}")
            # Create a copy of the instance with the same schedule
            sim_instance = instance.copy()
            
            # Regenerate processing time variability with different random seed
            if sim_instance.processing_time_variability > 0:
                print(f"    Applying processing time variability: {sim_instance.processing_time_variability}")
                sim_instance.set_processing_time_variability(sim_instance.processing_time_variability)
            
            # Regenerate machine breakdowns with different random seed
            has_breakdowns = False
            for machine in sim_instance.machines:
                if machine.failure_rate > 0:
                    has_breakdowns = True
                    machine.breakdown_times = []
            
            if has_breakdowns:
                print("    Regenerating machine breakdowns")
                sim_horizon = instance.makespan() * 2  # Use a reasonable horizon
                sim_instance.generate_machine_breakdowns(sim_horizon)
            
            # Re-simulate the schedule with uncertainty
            print("    Simulating schedule with uncertainty")
            self.simulate_schedule(sim_instance)
            
            # Record results
            sim_makespan = sim_instance.makespan()
            sim_flow_time = sim_instance.total_flow_time()
            makespans.append(sim_makespan)
            flow_times.append(sim_flow_time)
            print(f"    Simulation {i+1} results: makespan={sim_makespan}, flow_time={sim_flow_time}")
        
        # Calculate statistics
        avg_makespan = sum(makespans) / len(makespans)
        std_makespan = statistics.stdev(makespans) if len(makespans) > 1 else 0
        
        avg_flow_time = sum(flow_times) / len(flow_times)
        std_flow_time = statistics.stdev(flow_times) if len(flow_times) > 1 else 0
        
        # Calculate robust fitness (lower is better)
        robust_fitness = avg_makespan + self.stability_weight * std_makespan
        print(f"    Uncertainty evaluation complete: avg_makespan={avg_makespan}, std_dev={std_makespan}, robust_fitness={robust_fitness}")
        
        return {
            "makespan": avg_makespan,
            "makespan_std": std_makespan,
            "total_flow_time": avg_flow_time,
            "flow_time_std": std_flow_time,
            "robust_fitness": robust_fitness,
            "makespans": makespans,
            "flow_times": flow_times
        }
    
    def simulate_schedule(self, instance: JSSPInstance) -> None:
        """
        Simulate a schedule with uncertainty (processing time variability and machine breakdowns).
        
        Args:
            instance: JSSP instance with an initial schedule
        """
        # This would run an event-driven simulation to account for:
        # 1. Job arrival times
        # 2. Processing time variability
        # 3. Machine breakdowns
        
        # Implementation will be provided in the specific scheduler classes
        pass 