# Configuration for Edge-Aware Federated Learning System

import os
from typing import Dict, List, Any


class ConfigSection:
    """Base class for configuration sections"""
    def __init__(self, config_dict: Dict[str, Any]):
        self._data = {}
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key.upper(), ConfigSection(value))
                self._data[key.upper()] = getattr(self, key.upper())
            else:
                setattr(self, key.upper(), value)
                self._data[key.upper()] = value
    
    def get(self, key: str, default=None):
        """Get method for dictionary-like access"""
        # Try both uppercase and lowercase versions
        upper_key = key.upper()
        if hasattr(self, upper_key):
            return getattr(self, upper_key)
        if hasattr(self, key):
            return getattr(self, key)
        return default
    
    def __getitem__(self, key: str):
        """Dictionary-style access"""
        upper_key = key.upper()
        if hasattr(self, upper_key):
            return getattr(self, upper_key)
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Key '{key}' not found")
    
    def __contains__(self, key: str):
        """Check if key exists"""
        return hasattr(self, key.upper()) or hasattr(self, key)
    
    def keys(self):
        """Return dictionary keys"""
        return self._data.keys()
    
    def values(self):
        """Return dictionary values"""
        return self._data.values()
    
    def items(self):
        """Return dictionary items"""
        return self._data.items()
    
    def __dict__(self):
        """Return as dictionary"""
        return self._data

class Config:
    """Main configuration class for the Edge-Aware FL system"""
    
    def __init__(self):
        # Network configuration
        self.NETWORK = ConfigSection({
            "BASE_LATENCY": 0.05,  # 5G typical latency in seconds
            "BANDWIDTH": 100,      # Mbps
            "PACKET_LOSS_RATE": 0.001,
            "JITTER_MS": 0.5,
            "NUM_EDGE_SERVERS": 3,
            "DISCOVERY_INTERVAL": 30,
            "MAX_DISCOVERY_RANGE": 1000,
            "SIGNAL_THRESHOLD": -80
        })
        
        # Edge server configuration
        self.EDGE = ConfigSection({
            "MAX_CLIENTS": 50,
            "CPU_CORES": 4,
            "MEMORY_GB": 8,
            "STORAGE_GB": 64,
            "MIN_CLIENTS": 2,
            "MAX_WAIT_TIME": 120,
            "AGGREGATION_ALGORITHM": "fedavg",
            "CONTAINER_RUNTIME": "docker",
            "AUTO_SCALING": True
        })
        
        # Wearable device configuration
        self.DEVICES = ConfigSection({
            "CPU_CORES": 2,
            "MEMORY_MB": 512,
            "STORAGE_MB": 4096,
            "BATTERY_CAPACITY": 300,  # mAh
            "SENSOR_SAMPLING_RATE": 1,  # Hz
            "DATA_WINDOW_SIZE": 100,
            "MOBILITY_SPEED": 1.4,  # m/s
            "POWER_CONSUMPTION_BASE": 50,  # mW
            "POWER_CONSUMPTION_TRAINING": 200,  # mW
            "POWER_CONSUMPTION_COMMUNICATION": 100  # mW
        })
        
        # Federated learning configuration
        self.FL = ConfigSection({
            "NUM_ROUNDS": 50,
            "CLIENTS_PER_ROUND": 10,
            "LOCAL_EPOCHS": 5,
            "BATCH_SIZE": 8,  # Reduced from 32 to 8 for faster testing
            "LEARNING_RATE": 0.001,
            "OPTIMIZER": "adam",
            "MODEL_TYPE": "hybrid",  # 'cnn', 'lstm', 'hybrid'
            "INPUT_FEATURES": 8,  # Updated to match actual features from data processor
            "HIDDEN_UNITS": 64,
            "NUM_CLASSES": 2,
            "DROPOUT_RATE": 0.3,
            "MIN_FIT_CLIENTS": 5,
            "MIN_EVALUATE_CLIENTS": 5,
            "MIN_AVAILABLE_CLIENTS": 10,
            "FRACTION_FIT": 0.8,
            "FRACTION_EVALUATE": 0.2,
            "AGGREGATION": "fedavg"
        })
        
        # Data configuration
        self.DATA = ConfigSection({
            "PHYSIONET_PATH": "data/physionet",
            "SYNTHETIC_PATH": "data/synthetic",
            "NUM_USERS": 100,
            "DURATION_HOURS": 24,
            "ANOMALY_RATE": 0.1,
            "NORMALIZATION": "z_score",
            "WINDOW_SIZE": 60,
            "OVERLAP": 0.5,
            "FEATURE_EXTRACTION": ["statistical", "frequency"]
        })
        
        # Evaluation configuration
        self.EVALUATION = ConfigSection({
            "METRICS": ["accuracy", "precision", "recall", "f1_score", "auc_roc"],
            "PERFORMANCE_METRICS": ["training_time", "communication_cost", "energy_consumption"],
            "BENCHMARKS": ["centralized_learning", "local_only_learning", "traditional_federated"]
        })
        
        # Security configuration
        self.SECURITY = ConfigSection({
            "ENCRYPTION_ALGORITHM": "AES-256",
            "KEY_EXCHANGE": "ECDH",
            "DIGITAL_SIGNATURE": "ECDSA",
            "DIFFERENTIAL_PRIVACY": True,
            "NOISE_MULTIPLIER": 1.1,
            "MAX_GRAD_NORM": 1.0
        })
        
        # Simulation configuration
        self.SIMULATION = ConfigSection({
            "NUM_DEVICES": 10,
            "SIMULATION_TIME": 3600,  # seconds
            "REAL_TIME": False,
            "MOBILITY_ENABLED": True,
            "NETWORK_FAILURES": False,
            "DEVICE_FAILURES": False
        })
        
        # Logging configuration
        self.LOGGING = ConfigSection({
            "LEVEL": "INFO",
            "FORMAT": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "FILE": "logs/edge_fl_system.log",
            "MAX_SIZE_MB": 100,
            "BACKUP_COUNT": 5
        })


# Network Configuration
NETWORK_CONFIG = {
    "5g_simulation": {
        "bandwidth_mbps": 1000,  # 5G bandwidth
        "latency_ms": 1,         # Ultra-low latency
        "packet_loss": 0.001,    # Very low packet loss
        "jitter_ms": 0.5
    },
    "edge_discovery": {
        "discovery_interval": 30,  # seconds
        "max_discovery_range": 1000,  # meters
        "signal_threshold": -80  # dBm
    }
}
