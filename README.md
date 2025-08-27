# Heart Rate Monitor with HRV Analysis

A Python-based heart rate monitoring application with comprehensive Heart Rate Variability (HRV) analysis. Features multiple input methods: webcam, simulated data, and manual input.

This demo was written by AI.

## Prerequisites

- Python 3.8+
- [UV](https://docs.astral.sh/uv/) for dependency management

## Setup

1. Clone the repository:

   ```bash
   git clone <your-repo-url>
   cd heartrate
   ```

2. Install dependencies using UV:

   ```bash
   # Install UV if not already installed
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Create a virtual environment
   uv venv

   # Activate the virtual environment
   source .venv/bin/activate  # Unix
   .venv\Scripts\activate     # Windows

   # Install project dependencies
   uv pip install -r requirements.txt
   ```

## Running the Application

The Heart Rate Monitor supports three monitoring methods:

### 1. Simulated Mode (Default) - With HRV

```bash
python main.py --method simulate
```

Generates realistic heart rate data with physiological variability including:

- Respiratory Sinus Arrhythmia (RSA)
- Autonomic nervous system fluctuations
- Random physiological changes

### 2. Webcam Mode

```bash
python main.py --method webcam
```

Note: Requires a webcam and good lighting. Uses face detection and photoplethysmography.

### 3. Manual Input Mode

```bash
python main.py --method manual
```

Allows manual entry of heart rate values for testing and analysis.

### Command-line Options

- `--method`: Choose monitoring method (`webcam`, `simulate`, `manual`)
- `--window`: Set data window size (default: 300 readings)
- `--duration`: Simulation duration in seconds (default: 60)

## Heart Rate Variability (HRV) Features

The application calculates comprehensive HRV metrics from R-R intervals:

### Time-Domain Metrics

- **SDNN** (Standard Deviation of NN Intervals): Measures overall HRV
  - Normal range: 100-200ms for healthy adults
  - Higher values indicate better cardiovascular health
- **RMSSD** (Root Mean Square of Successive Differences): Short-term HRV
  - Reflects parasympathetic nervous system activity
- **pNN50**: Percentage of NN intervals differing by >50ms
  - Indicates autonomic nervous system flexibility

### Real-time Analysis

- HRV metrics calculated and displayed during monitoring
- Visual representation with heart rate zones
- Statistical summary with distribution analysis
- Professional-grade visualization suitable for research

### Example Usage

```bash
# Run 30-second HRV analysis
uv run python main.py --method simulate --duration 30

# Extended monitoring for better HRV accuracy
uv run python main.py --method simulate --duration 300
```

## Example Jupyter Notebook

See `heart_rate_demo.ipynb` for a comprehensive example of using the Heart Rate Monitor.

### Running the Notebook

```bash
# Install Jupyter if not already installed
uv pip install jupyter

# Launch Jupyter Notebook
jupyter notebook heart_rate_demo.ipynb
```

## Dependencies

- **OpenCV**: Camera input and face detection
- **Matplotlib**: Real-time visualization and plotting
- **NumPy**: Numerical computations and array operations
- **SciPy**: Signal processing for HRV analysis
- **Jupyter**: Interactive notebook support

## Output and Visualization

The application generates:

- **Real-time heart rate plot** with time series data
- **Heart rate distribution histogram**
- **HRV metrics display** with SDNN, RMSSD, and pNN50
- **Heart rate zones** (Resting, Fat Burn, Cardio, Peak)
- **Statistical summary** with averages and ranges
- **Saved plot** as `x.png` for further analysis

## Troubleshooting

- Ensure all required packages are installed with `uv pip install -r requirements.txt`
- For webcam mode, check camera permissions and lighting conditions
- HRV metrics require at least 10 R-R intervals for calculation
- For accurate HRV analysis, use longer monitoring periods (>5 minutes recommended)
- Minimum Python version: 3.8
