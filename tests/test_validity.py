"""
Regression tests for the Job Shop Scheduling Problem simulation.
"""

import pytest
import random
from jssp.data import JSSPInstance
from jssp.schedulers.genetic import precedence_preserving_crossover, Chromosome

def test_ppx_preserves_precedence():
    """
    Test that precedence_preserving_crossover preserves precedence constraints.
    """
    # Create a test instance
    instance = JSSPInstance.generate_random_instance(
        num_jobs=5,
        num_machines=3,
        min_proc_time=1,
        max_proc_time=10,
        seed=42
    )
    
    # Create two parent chromosomes
    parent1 = Chromosome(instance)
    parent2 = Chromosome(instance)
    
    # Perform crossover
    child = precedence_preserving_crossover(parent1, parent2)
    
    # Verify that the child sequence preserves precedence constraints
    assert instance.is_valid_sequence(child.sequence) 