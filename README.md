# Edge-Aware Federated Learning for Health Monitoring

A simulation platform for federated learning on wearable health devices with edge computing infrastructure.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the System
```bash
# Interactive launcher (recommended)
python launcher.py

# Command line interface
python main.py --devices 5 --servers 2 --rounds 3
```

## How to Use

### Interactive Menu
Run `python launcher.py` and choose from:
- **Quick Demo**: 5-minute demonstration
- **Small Experiment**: 5 devices, 10 rounds (~2 minutes)
- **Medium Experiment**: 20 devices, 50 rounds (~10 minutes)
- **Custom Experiment**: Define your own parameters
- **Visualize Results**: Generate plots from experiments

### Command Line Options
```bash
python main.py [OPTIONS]

Options:
  --devices INTEGER    Number of simulated wearable devices (default: 5)
  --servers INTEGER    Number of edge servers (default: 2)
  --rounds INTEGER     Number of FL training rounds (default: 3)
  --duration FLOAT     Experiment duration in hours (default: 0.25)
```

### Example Commands
```bash
# Quick test
python main.py --devices 3 --servers 1 --rounds 5

# Medium experiment
python main.py --devices 10 --servers 2 --rounds 20

# Large experiment
python main.py --devices 50 --servers 5 --rounds 100
```

## What the System Does

This platform simulates:
- **Wearable Devices**: Android smartwatches with health sensors (heart rate, accelerometer, gyroscope)
- **Edge Servers**: Distributed servers that perform federated learning aggregation
- **5G Network**: Realistic network conditions with signal strength and latency modeling
- **Federated Learning**: Privacy-preserving machine learning without sharing raw data

## System Architecture

```
[Wearable Devices] ↔ [Edge Servers] ↔ [FL Coordinator]
        ↓                   ↓              ↓
    Sensor Data      Model Aggregation   Experiment
    Local Training   Load Balancing      Coordination
```

## Results and Visualization

- **Experiment Results**: Saved in `results/` directory as JSON files
- **Visualizations**: Generated in `visualizations/` directory
- **Logs**: Detailed execution logs in `logs/` directory

To generate visualizations after running experiments:
```bash
python launcher.py
# Select option 6: "Visualize Results"
```

## Configuration

Modify system behavior in `config/config.py`:
```python
# Federated Learning settings
FL.BATCH_SIZE = 8          # Training batch size
FL.LOCAL_EPOCHS = 5        # Local training epochs
FL.LEARNING_RATE = 0.001   # Learning rate

# Network settings
NETWORK.SIGNAL_THRESHOLD = -80.0   # Minimum signal strength (dBm)
NETWORK.BANDWIDTH_MBPS = 1000      # Network bandwidth

# Device settings
DEVICES.BATTERY_CAPACITY = 300     # Battery capacity (mAh)
```

## Requirements

- **Python**: 3.8 or higher
- **Dependencies**: Listed in `requirements.txt`
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 2GB available space

## Key Features

✅ **Fully Working System**: All components tested and functional  
✅ **Realistic Simulation**: 5G network modeling and device mobility  
✅ **Multiple ML Models**: CNN, LSTM, and hybrid architectures  
✅ **Privacy-Preserving**: Raw data never leaves devices  
✅ **Scalable**: Supports 100+ simulated devices  
✅ **Comprehensive Metrics**: Performance tracking and visualization  

## Troubleshooting

### Common Issues

**"No module named 'torch'"**
```bash
pip install torch
```

**"No edge servers ready for round 0"**
- Reduce number of devices or increase experiment duration
- Check that MIN_CLIENTS setting in config matches your setup

**Unicode encoding errors**
- Run: `chcp 65001` before launching (Windows)
- Or use the interactive launcher which handles encoding automatically

**Import errors**
- Ensure you're running from the project root directory
- Check that all dependencies are installed: `pip install -r requirements.txt`

### Getting Help

1. Check the logs in `logs/` directory for detailed error information
2. Review the configuration in `config/config.py`
3. See `CODE_DOCUMENTATION.txt` for technical details
4. See `SYSTEM_REPORT.txt` for research paper and comprehensive documentation

## Performance Tips

**For faster testing:**
```python
# In config/config.py
FL.BATCH_SIZE = 4        # Reduce from 8
FL.LOCAL_EPOCHS = 2      # Reduce from 5
```

**For realistic experiments:**
```python
FL.BATCH_SIZE = 32       # Increase for production
DEVICES.SENSOR_FREQUENCY = 10.0  # Higher sampling rate
```

## Docker Deployment

```bash
# Build and run with Docker
cd docker
docker-compose up

# Or run individual components
docker build -t edge-fl .
docker run -p 8080:8080 edge-fl
```

---

**Status**: ✅ Fully Working System  
**License**: Educational and Research Use  
**Version**: 1.0
