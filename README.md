# Job Shop Scheduling Problem (JSSP) Simulation Framework

This project implements a simulation framework for solving Job Shop Scheduling Problems using Genetic Algorithms (GA) and comparing with simpler scheduling rules like FIFO and SPT.

## Project Structure
- `jssp/` - Core simulation framework
  - `data.py` - Job and machine data structures
  - `schedulers/` - Different scheduling algorithms
    - `genetic.py` - Genetic Algorithm implementation
    - `simple.py` - Simple rules like FIFO and SPT
  - `visualization.py` - Visualization utilities
- `app_streamlit.py` - Streamlit web application with interactive UI

## Setup
```bash
pip install -r requirements.txt
```

## Running the simulation
```bash
python -m jssp.main
```

## Running the Web UI
```bash
streamlit run app_streamlit.py
```

## Features
- Interactive problem configuration with random or custom instances
- Multiple scheduling algorithms (FIFO, SPT, and Genetic Algorithm)
- Visualization of schedules using Gantt charts
- Performance metrics comparison
- Genetic Algorithm parameter tuning
- Detailed explanations of scheduling concepts and terminology 