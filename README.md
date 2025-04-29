# Job Shop Scheduling Problem (JSSP) Simulation Framework

This project implements a simulation framework for solving Job Shop Scheduling Problems using Genetic Algorithms (GA) and comparing with simpler scheduling rules like FIFO and SPT.

## Project Structure
- `jssp/` - Core simulation framework
  - `data.py` - Job and machine data structures
  - `schedulers/` - Different scheduling algorithms
    - `genetic.py` - Genetic Algorithm implementation
    - `simple.py` - Simple rules like FIFO and SPT
  - `visualization.py` - Visualization utilities
- `app_qt.py` - PyQt5 desktop application for UI

## Setup
```bash
pip install -r requirements.txt
```

## Running the simulation
```bash
python -m jssp.main
```

## Running the Desktop UI
```bash
python app_qt.py
``` 