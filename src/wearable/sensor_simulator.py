"""
Sensor Simulator for Wearable Devices
Simulates physiological sensors (heart rate, SpO2, accelerometer)
"""

import asyncio
import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import numpy as np
import math

from ..utils.logger import setup_logger


@dataclass
class SensorReading:
    """Individual sensor reading"""
    sensor_type: str
    value: float
    unit: str
    timestamp: datetime
    accuracy: float
    noise_level: float


@dataclass
class SensorConfig:
    """Sensor configuration parameters"""
    sampling_rate_hz: float
    accuracy: float
    power_consumption_mw: float
    noise_std: float = 0.05
    drift_rate: float = 0.001


class PhysiologicalModel:
    """
    Models realistic physiological patterns and variations
    """
    
    def __init__(self, user_profile: Dict[str, Any] = None):
        self.user_profile = user_profile or {}
        
        # User characteristics
        self.age = self.user_profile.get("age", 30)
        self.fitness_level = self.user_profile.get("fitness_level", 0.7)  # 0-1
        self.health_status = self.user_profile.get("health_status", "healthy")
        
        # Baseline physiological parameters
        self.baseline_hr = self._calculate_baseline_hr()
        self.baseline_spo2 = self._calculate_baseline_spo2()
        
        # Activity state
        self.current_activity = "resting"  # resting, walking, running, sleeping
        self.activity_intensity = 0.0  # 0-1
        
        # Circadian rhythm parameters
        self.circadian_phase = 0.0  # 0-24 hours
        
        # Health anomaly simulation
        self.anomaly_probability = 0.05  # 5% chance of anomaly
        self.current_anomaly = None
        
    def _calculate_baseline_hr(self) -> float:
        """Calculate baseline heart rate based on user profile"""
        # Age-adjusted resting heart rate
        base_hr = 220 - self.age
        resting_hr = base_hr * 0.3 + (1 - self.fitness_level) * 20
        return max(50, min(100, resting_hr))
    
    def _calculate_baseline_spo2(self) -> float:
        """Calculate baseline SpO2 based on user profile"""
        base_spo2 = 98.5
        
        if self.health_status == "respiratory_condition":
            base_spo2 -= 3
        elif self.health_status == "cardiac_condition":
            base_spo2 -= 1
        
        return max(92, min(100, base_spo2))
    
    def update_activity(self, activity: str, intensity: float = 0.0):
        """Update current activity state"""
        self.current_activity = activity
        self.activity_intensity = max(0.0, min(1.0, intensity))
    
    def update_circadian_phase(self, hour_of_day: float):
        """Update circadian rhythm phase"""
        self.circadian_phase = hour_of_day % 24
    
    def generate_heart_rate(self) -> float:
        """Generate realistic heart rate reading"""
        hr = self.baseline_hr
        
        # Activity effect
        activity_multipliers = {
            "sleeping": 0.7,
            "resting": 1.0,
            "walking": 1.3,
            "running": 1.8,
            "exercising": 2.0
        }
        
        activity_mult = activity_multipliers.get(self.current_activity, 1.0)
        activity_mult += self.activity_intensity * 0.5
        hr *= activity_mult
        
        # Circadian rhythm effect
        circadian_effect = 0.1 * math.sin(2 * math.pi * (self.circadian_phase - 6) / 24)
        hr *= (1 + circadian_effect)
        
        # Add physiological variability
        hr += random.gauss(0, 5)  # Heart rate variability
        
        # Simulate anomalies
        if random.random() < self.anomaly_probability:
            anomaly_type = random.choice(["bradycardia", "tachycardia", "arrhythmia"])
            hr = self._apply_heart_rate_anomaly(hr, anomaly_type)
            self.current_anomaly = anomaly_type
        else:
            self.current_anomaly = None
        
        return max(30, min(220, hr))
    
    def generate_spo2(self) -> float:
        """Generate realistic SpO2 reading"""
        spo2 = self.baseline_spo2
        
        # Activity effect (slight decrease during intense exercise)
        if self.current_activity in ["running", "exercising"]:
            spo2 -= self.activity_intensity * 2
        
        # Add measurement noise
        spo2 += random.gauss(0, 0.5)
        
        # Simulate hypoxemia events
        if random.random() < 0.02:  # 2% chance
            spo2 -= random.uniform(5, 15)
            self.current_anomaly = "hypoxemia"
        
        return max(80, min(100, spo2))
    
    def generate_accelerometer(self) -> tuple:
        """Generate realistic accelerometer readings (x, y, z in g)"""
        # Base gravity component
        base_x, base_y, base_z = 0, 0, 1  # Device oriented flat
        
        # Activity-based movement patterns
        movement_patterns = {
            "sleeping": (0.02, 0.02, 0.02),
            "resting": (0.05, 0.05, 0.05),
            "walking": (0.3, 0.2, 0.4),
            "running": (0.8, 0.6, 1.2),
            "exercising": (1.0, 1.0, 1.0)
        }
        
        std_x, std_y, std_z = movement_patterns.get(self.current_activity, (0.1, 0.1, 0.1))
        
        # Scale by activity intensity
        std_x *= (1 + self.activity_intensity)
        std_y *= (1 + self.activity_intensity)
        std_z *= (1 + self.activity_intensity)
        
        # Generate readings
        acc_x = base_x + random.gauss(0, std_x)
        acc_y = base_y + random.gauss(0, std_y)
        acc_z = base_z + random.gauss(0, std_z)
        
        return acc_x, acc_y, acc_z
    
    def _apply_heart_rate_anomaly(self, normal_hr: float, anomaly_type: str) -> float:
        """Apply heart rate anomaly patterns"""
        if anomaly_type == "bradycardia":
            return min(normal_hr, random.uniform(30, 50))
        elif anomaly_type == "tachycardia":
            return max(normal_hr, random.uniform(120, 180))
        elif anomaly_type == "arrhythmia":
            # Irregular rhythm
            variation = random.uniform(0.7, 1.3)
            return normal_hr * variation
        
        return normal_hr


class SensorSimulator:
    """
    Simulates multiple physiological sensors on a wearable device
    """
    
    def __init__(self, sensor_configs: Dict[str, Dict[str, Any]] = None):
        self.logger = setup_logger("SensorSimulator")
        
        # Default sensor configurations
        default_configs = {
            "heart_rate": {
                "sampling_rate_hz": 1.0,
                "accuracy": 0.95,
                "power_consumption_mw": 5.0,
                "noise_std": 2.0
            },
            "spo2": {
                "sampling_rate_hz": 0.2,
                "accuracy": 0.92,
                "power_consumption_mw": 8.0,
                "noise_std": 0.5
            },
            "accelerometer": {
                "sampling_rate_hz": 50.0,
                "accuracy": 0.98,
                "power_consumption_mw": 3.0,
                "noise_std": 0.02
            }
        }
        
        # Merge with provided configs
        self.sensor_configs = {}
        for sensor_type, default_config in default_configs.items():
            config = sensor_configs.get(sensor_type, {}) if sensor_configs else {}
            self.sensor_configs[sensor_type] = {**default_config, **config}
        
        # Initialize physiological model
        self.physio_model = PhysiologicalModel()
        
        # Sensor state
        self.last_readings = {}
        self.sensor_drift = {}
        self.calibration_factors = {}
        
        # Initialize sensor state
        for sensor_type in self.sensor_configs:
            self.sensor_drift[sensor_type] = 1.0
            self.calibration_factors[sensor_type] = random.uniform(0.98, 1.02)
        
        self.logger.info("Sensor simulator initialized")
    
    async def collect_readings(self) -> Dict[str, SensorReading]:
        """
        Collect readings from all sensors
        
        Returns:
            Dictionary of sensor readings
        """
        readings = {}
        current_time = datetime.now()
        
        # Update physiological model based on time of day
        hour_of_day = current_time.hour + current_time.minute / 60
        self.physio_model.update_circadian_phase(hour_of_day)
        
        # Simulate activity detection from accelerometer
        if "accelerometer" in self.sensor_configs:
            acc_reading = await self._read_accelerometer()
            readings["accelerometer"] = acc_reading
            
            # Infer activity from accelerometer data
            magnitude = math.sqrt(sum(x**2 for x in [acc_reading.value]))
            if magnitude < 1.1:
                activity = "resting"
                intensity = 0.0
            elif magnitude < 1.5:
                activity = "walking"
                intensity = 0.3
            else:
                activity = "running"
                intensity = min(1.0, (magnitude - 1.5) / 2.0)
            
            self.physio_model.update_activity(activity, intensity)
        
        # Collect other sensor readings
        if "heart_rate" in self.sensor_configs:
            readings["heart_rate"] = await self._read_heart_rate()
        
        if "spo2" in self.sensor_configs:
            readings["spo2"] = await self._read_spo2()
        
        return readings
    
    async def _read_heart_rate(self) -> SensorReading:
        """Read heart rate sensor"""
        config = self.sensor_configs["heart_rate"]
        
        # Generate physiological reading
        true_hr = self.physio_model.generate_heart_rate()
        
        # Apply sensor characteristics
        measured_hr = self._apply_sensor_effects(true_hr, "heart_rate", config)
        
        return SensorReading(
            sensor_type="heart_rate",
            value=measured_hr,
            unit="bpm",
            timestamp=datetime.now(),
            accuracy=config["accuracy"],
            noise_level=config["noise_std"]
        )
    
    async def _read_spo2(self) -> SensorReading:
        """Read SpO2 sensor"""
        config = self.sensor_configs["spo2"]
        
        # Generate physiological reading
        true_spo2 = self.physio_model.generate_spo2()
        
        # Apply sensor characteristics
        measured_spo2 = self._apply_sensor_effects(true_spo2, "spo2", config)
        
        return SensorReading(
            sensor_type="spo2",
            value=measured_spo2,
            unit="%",
            timestamp=datetime.now(),
            accuracy=config["accuracy"],
            noise_level=config["noise_std"]
        )
    
    async def _read_accelerometer(self) -> SensorReading:
        """Read accelerometer sensor"""
        config = self.sensor_configs["accelerometer"]
        
        # Generate accelerometer readings
        acc_x, acc_y, acc_z = self.physio_model.generate_accelerometer()
        
        # Calculate magnitude
        magnitude = math.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        
        # Apply sensor effects
        measured_magnitude = self._apply_sensor_effects(magnitude, "accelerometer", config)
        
        return SensorReading(
            sensor_type="accelerometer",
            value=measured_magnitude,
            unit="g",
            timestamp=datetime.now(),
            accuracy=config["accuracy"],
            noise_level=config["noise_std"]
        )
    
    def _apply_sensor_effects(
        self, 
        true_value: float, 
        sensor_type: str, 
        config: Dict[str, Any]
    ) -> float:
        """
        Apply realistic sensor effects (noise, drift, calibration errors)
        """
        measured_value = true_value
        
        # Apply calibration factor
        measured_value *= self.calibration_factors[sensor_type]
        
        # Apply sensor drift (slow degradation over time)
        drift_rate = config.get("drift_rate", 0.001)
        self.sensor_drift[sensor_type] *= (1 + random.gauss(0, drift_rate))
        measured_value *= self.sensor_drift[sensor_type]
        
        # Add measurement noise
        noise_std = config["noise_std"]
        measured_value += random.gauss(0, noise_std)
        
        # Apply accuracy limitations (systematic errors)
        accuracy = config["accuracy"]
        if random.random() > accuracy:
            # Introduce measurement error
            error_magnitude = random.uniform(0.05, 0.15) * true_value
            measured_value += random.choice([-1, 1]) * error_magnitude
        
        return measured_value
    
    def simulate_sensor_failure(self, sensor_type: str, failure_type: str = "noise"):
        """
        Simulate sensor failure modes
        
        Args:
            sensor_type: Type of sensor to affect
            failure_type: "noise", "drift", "offset", "dead"
        """
        if sensor_type not in self.sensor_configs:
            return
        
        if failure_type == "noise":
            self.sensor_configs[sensor_type]["noise_std"] *= 5
        elif failure_type == "drift":
            self.sensor_drift[sensor_type] *= random.uniform(0.8, 1.2)
        elif failure_type == "offset":
            self.calibration_factors[sensor_type] *= random.uniform(0.9, 1.1)
        elif failure_type == "dead":
            self.sensor_configs[sensor_type]["accuracy"] = 0.0
        
        self.logger.warning(f"Simulated {failure_type} failure in {sensor_type} sensor")
    
    def get_sensor_status(self) -> Dict[str, Dict[str, Any]]:
        """Get current sensor status and health"""
        status = {}
        
        for sensor_type, config in self.sensor_configs.items():
            status[sensor_type] = {
                "accuracy": config["accuracy"],
                "noise_level": config["noise_std"],
                "drift_factor": self.sensor_drift.get(sensor_type, 1.0),
                "calibration_factor": self.calibration_factors.get(sensor_type, 1.0),
                "power_consumption_mw": config["power_consumption_mw"],
                "sampling_rate_hz": config["sampling_rate_hz"],
                "last_reading": self.last_readings.get(sensor_type)
            }
        
        return status
