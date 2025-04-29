"""
Scheduling algorithms for Job Shop Scheduling Problems.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

from jssp.data import JSSPInstance

class Scheduler(ABC):
    """Base class for all scheduling algorithms."""
    
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