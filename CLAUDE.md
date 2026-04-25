# LERNA Project

## Project Structure
- `experiments/` - Experiment code and results
- `scripts/` - Utility scripts  
- `configs/` - Configuration files
- `tests/` - Test files

## Key Files
- `add_bulk_data.py` - Add bulk data to experiments
- `add_data.py` - Add individual data records
- `update_records.py` - Update existing records
- `filter_records.py` - Filter experiment data
- `run_slurm.sh` - SLURM job runner

## Technologies
- Python 3.x
- SLURM for HPC workloads
- Docker for containerization

## Commands
- Run experiments: `./run_slurm.sh`
- Setup: `python setup.py`
- Tests: `pytest tests/`
