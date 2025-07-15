"""
Health Data Processor for Wearable Devices
Processes raw sensor data into features for ML models
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import math
from scipy import signal
from scipy.stats import entropy
import warnings

from ..utils.logger import setup_logger


@dataclass
class ProcessedHealthData:
    """Processed health data point"""
    timestamp: datetime
    features: List[float]
    feature_names: List[str]
    label: Optional[int] = None  # 0: normal, 1: anomaly
    confidence: float = 1.0
    raw_data: Dict[str, Any] = None


class HealthDataProcessor:
    """
    Processes physiological sensor data for machine learning
    """
    
    def __init__(self, window_size_seconds: int = 60, overlap: float = 0.5):
        """
        Initialize data processor
        
        Args:
            window_size_seconds: Size of analysis window
            overlap: Overlap between windows (0-1)
        """
        self.logger = setup_logger("HealthDataProcessor")
        self.window_size = window_size_seconds
        self.overlap = overlap
        
        # Data buffers for windowed analysis
        self.data_buffer = {
            "heart_rate": [],
            "spo2": [],
            "accelerometer": [],
            "timestamps": []
        }
        
        # Feature extraction parameters
        self.feature_extractors = {
            "statistical": self._extract_statistical_features,
            "frequency": self._extract_frequency_features,
            "temporal": self._extract_temporal_features,
            "physiological": self._extract_physiological_features
        }
        
        # Anomaly detection thresholds
        self.anomaly_thresholds = {
            "heart_rate_low": 50,
            "heart_rate_high": 120,
            "spo2_low": 95,
            "high_activity_threshold": 2.0  # g
        }
        
        self.logger.info("Health data processor initialized")
    
    def process_realtime_data(self, sensor_readings: Dict[str, Any]) -> ProcessedHealthData:
        """
        Process real-time sensor readings
        
        Args:
            sensor_readings: Dictionary of sensor readings
            
        Returns:
            Processed health data point
        """
        current_time = datetime.now()
        
        # Extract values from sensor readings
        hr_value = None
        spo2_value = None
        acc_value = None
        
        for sensor_type, reading in sensor_readings.items():
            if hasattr(reading, 'value'):
                if sensor_type == "heart_rate":
                    hr_value = reading.value
                elif sensor_type == "spo2":
                    spo2_value = reading.value
                elif sensor_type == "accelerometer":
                    acc_value = reading.value
        
        # Update data buffers
        self._update_buffers(current_time, hr_value, spo2_value, acc_value)
        
        # Extract features from current readings
        features = []
        feature_names = []
        
        # Instantaneous features
        if hr_value is not None:
            features.extend([hr_value, self._normalize_heart_rate(hr_value)])
            feature_names.extend(["heart_rate", "heart_rate_normalized"])
        
        if spo2_value is not None:
            features.extend([spo2_value, self._normalize_spo2(spo2_value)])
            feature_names.extend(["spo2", "spo2_normalized"])
        
        if acc_value is not None:
            features.extend([acc_value, self._categorize_activity(acc_value)])
            feature_names.extend(["accelerometer_magnitude", "activity_level"])
        
        # Add time-based features
        hour_sin = math.sin(2 * math.pi * current_time.hour / 24)
        hour_cos = math.cos(2 * math.pi * current_time.hour / 24)
        features.extend([hour_sin, hour_cos])
        feature_names.extend(["hour_sin", "hour_cos"])
        
        # Perform anomaly detection
        label = self._detect_anomaly(hr_value, spo2_value, acc_value)
        confidence = self._calculate_confidence(features)
        
        return ProcessedHealthData(
            timestamp=current_time,
            features=features,
            feature_names=feature_names,
            label=label,
            confidence=confidence,
            raw_data=sensor_readings
        )
    
    def process_windowed_data(self) -> Optional[ProcessedHealthData]:
        """
        Process data using sliding window approach
        
        Returns:
            Processed data with windowed features, or None if insufficient data
        """
        if not self._has_sufficient_data():
            return None
        
        current_time = datetime.now()
        
        # Get data within current window
        window_data = self._get_window_data()
        
        if not window_data:
            return None
        
        # Extract comprehensive features
        features = []
        feature_names = []
        
        # Statistical features
        stat_features, stat_names = self._extract_statistical_features(window_data)
        features.extend(stat_features)
        feature_names.extend(stat_names)
        
        # Frequency domain features
        freq_features, freq_names = self._extract_frequency_features(window_data)
        features.extend(freq_features)
        feature_names.extend(freq_names)
        
        # Temporal features
        temp_features, temp_names = self._extract_temporal_features(window_data)
        features.extend(temp_features)
        feature_names.extend(temp_names)
        
        # Physiological features
        physio_features, physio_names = self._extract_physiological_features(window_data)
        features.extend(physio_features)
        feature_names.extend(physio_names)
        
        # Anomaly detection on windowed data
        label = self._detect_windowed_anomaly(window_data)
        confidence = self._calculate_confidence(features)
        
        return ProcessedHealthData(
            timestamp=current_time,
            features=features,
            feature_names=feature_names,
            label=label,
            confidence=confidence,
            raw_data=window_data
        )
    
    def _update_buffers(self, timestamp: datetime, hr: float, spo2: float, acc: float):
        """Update data buffers with new readings"""
        self.data_buffer["timestamps"].append(timestamp)
        self.data_buffer["heart_rate"].append(hr if hr is not None else np.nan)
        self.data_buffer["spo2"].append(spo2 if spo2 is not None else np.nan)
        self.data_buffer["accelerometer"].append(acc if acc is not None else np.nan)
        
        # Maintain buffer size (keep last 5 minutes of data)
        max_size = 300  # 5 minutes at 1Hz
        for key in self.data_buffer:
            if len(self.data_buffer[key]) > max_size:
                self.data_buffer[key] = self.data_buffer[key][-max_size:]
    
    def _has_sufficient_data(self) -> bool:
        """Check if there's sufficient data for windowed analysis"""
        return len(self.data_buffer["timestamps"]) >= self.window_size
    
    def _get_window_data(self) -> Dict[str, List[float]]:
        """Get data within the current analysis window"""
        if not self._has_sufficient_data():
            return {}
        
        # Get data for the last window_size seconds
        end_idx = len(self.data_buffer["timestamps"])
        start_idx = max(0, end_idx - self.window_size)
        
        window_data = {}
        for key in ["heart_rate", "spo2", "accelerometer"]:
            window_data[key] = self.data_buffer[key][start_idx:end_idx]
        
        return window_data
    
    def _extract_statistical_features(self, data: Dict[str, List[float]]) -> Tuple[List[float], List[str]]:
        """Extract statistical features from windowed data"""
        features = []
        feature_names = []
        
        for signal_name, values in data.items():
            # Remove NaN values
            clean_values = [v for v in values if not np.isnan(v)]
            
            if len(clean_values) < 5:  # Insufficient data
                # Add placeholder features
                features.extend([0, 0, 0, 0, 0])
                feature_names.extend([
                    f"{signal_name}_mean", f"{signal_name}_std",
                    f"{signal_name}_min", f"{signal_name}_max",
                    f"{signal_name}_range"
                ])
                continue
            
            values_array = np.array(clean_values)
            
            # Basic statistics
            mean_val = np.mean(values_array)
            std_val = np.std(values_array)
            min_val = np.min(values_array)
            max_val = np.max(values_array)
            range_val = max_val - min_val
            
            features.extend([mean_val, std_val, min_val, max_val, range_val])
            feature_names.extend([
                f"{signal_name}_mean", f"{signal_name}_std",
                f"{signal_name}_min", f"{signal_name}_max",
                f"{signal_name}_range"
            ])
            
            # Advanced statistics
            if len(clean_values) >= 10:
                # Percentiles
                p25 = np.percentile(values_array, 25)
                p75 = np.percentile(values_array, 75)
                iqr = p75 - p25
                
                # Skewness and kurtosis (simplified)
                normalized = (values_array - mean_val) / (std_val + 1e-8)
                skewness = np.mean(normalized ** 3)
                kurtosis = np.mean(normalized ** 4) - 3
                
                features.extend([p25, p75, iqr, skewness, kurtosis])
                feature_names.extend([
                    f"{signal_name}_p25", f"{signal_name}_p75",
                    f"{signal_name}_iqr", f"{signal_name}_skewness",
                    f"{signal_name}_kurtosis"
                ])
        
        return features, feature_names
    
    def _extract_frequency_features(self, data: Dict[str, List[float]]) -> Tuple[List[float], List[str]]:
        """Extract frequency domain features"""
        features = []
        feature_names = []
        
        for signal_name, values in data.items():
            clean_values = [v for v in values if not np.isnan(v)]
            
            if len(clean_values) < 16:  # Insufficient for FFT
                features.extend([0, 0, 0, 0])
                feature_names.extend([
                    f"{signal_name}_dominant_freq", f"{signal_name}_spectral_energy",
                    f"{signal_name}_spectral_entropy", f"{signal_name}_power_ratio"
                ])
                continue
            
            values_array = np.array(clean_values)
            
            # Remove DC component
            values_array = values_array - np.mean(values_array)
            
            # Apply window to reduce spectral leakage
            windowed = values_array * np.hanning(len(values_array))
            
            # Compute FFT
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fft = np.fft.fft(windowed)
                freqs = np.fft.fftfreq(len(windowed), d=1.0)  # Assuming 1Hz sampling
                
                # Power spectral density
                psd = np.abs(fft) ** 2
                
                # Consider only positive frequencies
                pos_freqs = freqs[:len(freqs)//2]
                pos_psd = psd[:len(psd)//2]
                
                if np.sum(pos_psd) > 0:
                    # Dominant frequency
                    dominant_freq_idx = np.argmax(pos_psd)
                    dominant_freq = pos_freqs[dominant_freq_idx]
                    
                    # Spectral energy
                    spectral_energy = np.sum(pos_psd)
                    
                    # Spectral entropy
                    normalized_psd = pos_psd / (np.sum(pos_psd) + 1e-8)
                    spectral_entropy = entropy(normalized_psd + 1e-8)
                    
                    # Low frequency to high frequency power ratio
                    low_freq_mask = pos_freqs < 0.1  # Below 0.1 Hz
                    high_freq_mask = pos_freqs > 0.1
                    
                    low_power = np.sum(pos_psd[low_freq_mask])
                    high_power = np.sum(pos_psd[high_freq_mask])
                    power_ratio = low_power / (high_power + 1e-8)
                else:
                    dominant_freq = 0
                    spectral_energy = 0
                    spectral_entropy = 0
                    power_ratio = 0
            
            features.extend([dominant_freq, spectral_energy, spectral_entropy, power_ratio])
            feature_names.extend([
                f"{signal_name}_dominant_freq", f"{signal_name}_spectral_energy",
                f"{signal_name}_spectral_entropy", f"{signal_name}_power_ratio"
            ])
        
        return features, feature_names
    
    def _extract_temporal_features(self, data: Dict[str, List[float]]) -> Tuple[List[float], List[str]]:
        """Extract temporal features"""
        features = []
        feature_names = []
        
        for signal_name, values in data.items():
            clean_values = [v for v in values if not np.isnan(v)]
            
            if len(clean_values) < 5:
                features.extend([0, 0, 0])
                feature_names.extend([
                    f"{signal_name}_trend", f"{signal_name}_variability",
                    f"{signal_name}_zero_crossings"
                ])
                continue
            
            values_array = np.array(clean_values)
            
            # Linear trend (slope)
            x = np.arange(len(values_array))
            trend = np.polyfit(x, values_array, 1)[0]
            
            # Variability (coefficient of variation)
            mean_val = np.mean(values_array)
            std_val = np.std(values_array)
            variability = std_val / (abs(mean_val) + 1e-8)
            
            # Zero crossings (relative to mean)
            centered = values_array - mean_val
            zero_crossings = np.sum(np.diff(np.sign(centered)) != 0)
            
            features.extend([trend, variability, zero_crossings])
            feature_names.extend([
                f"{signal_name}_trend", f"{signal_name}_variability",
                f"{signal_name}_zero_crossings"
            ])
        
        return features, feature_names
    
    def _extract_physiological_features(self, data: Dict[str, List[float]]) -> Tuple[List[float], List[str]]:
        """Extract physiological domain-specific features"""
        features = []
        feature_names = []
        
        # Heart rate variability features
        if "heart_rate" in data:
            hr_values = [v for v in data["heart_rate"] if not np.isnan(v)]
            
            if len(hr_values) >= 10:
                hr_array = np.array(hr_values)
                
                # RR intervals (inverse of heart rate)
                rr_intervals = 60.0 / (hr_array + 1e-8)  # Convert to RR intervals in seconds
                
                # RMSSD (root mean square of successive differences)
                rr_diffs = np.diff(rr_intervals)
                rmssd = np.sqrt(np.mean(rr_diffs ** 2))
                
                # pNN50 (percentage of successive RR intervals differing by > 50ms)
                pnn50 = np.sum(np.abs(rr_diffs) > 0.05) / len(rr_diffs) * 100
                
                # Triangular index approximation
                tri_index = len(rr_intervals) / (np.max(rr_intervals) - np.min(rr_intervals) + 1e-8)
                
                features.extend([rmssd, pnn50, tri_index])
                feature_names.extend(["hrv_rmssd", "hrv_pnn50", "hrv_triangular_index"])
            else:
                features.extend([0, 0, 0])
                feature_names.extend(["hrv_rmssd", "hrv_pnn50", "hrv_triangular_index"])
        
        # SpO2 desaturation features
        if "spo2" in data:
            spo2_values = [v for v in data["spo2"] if not np.isnan(v)]
            
            if len(spo2_values) >= 5:
                spo2_array = np.array(spo2_values)
                
                # Desaturation events (drops below 95%)
                desaturation_events = np.sum(spo2_array < 95)
                
                # Mean desaturation depth
                desat_values = spo2_array[spo2_array < 95]
                mean_desat_depth = np.mean(95 - desat_values) if len(desat_values) > 0 else 0
                
                # Recovery time (simplified)
                recovery_time = 0
                if len(desat_values) > 0:
                    # Find recovery patterns
                    below_threshold = spo2_array < 95
                    transitions = np.diff(below_threshold.astype(int))
                    recovery_events = np.where(transitions == -1)[0]  # Transitions from below to above
                    if len(recovery_events) > 0:
                        recovery_time = len(recovery_events)
                
                features.extend([desaturation_events, mean_desat_depth, recovery_time])
                feature_names.extend(["spo2_desat_events", "spo2_desat_depth", "spo2_recovery_time"])
            else:
                features.extend([0, 0, 0])
                feature_names.extend(["spo2_desat_events", "spo2_desat_depth", "spo2_recovery_time"])
        
        # Activity-related features
        if "accelerometer" in data:
            acc_values = [v for v in data["accelerometer"] if not np.isnan(v)]
            
            if len(acc_values) >= 5:
                acc_array = np.array(acc_values)
                
                # Activity intensity
                activity_intensity = np.mean(np.abs(acc_array - 1.0))  # Deviation from gravity
                
                # Step count estimation (simplified)
                # Look for periodic patterns in accelerometer data
                step_candidates = np.where(np.diff(acc_array) > 0.2)[0]  # Threshold crossings
                estimated_steps = len(step_candidates) / 2  # Rough estimation
                
                # Movement variability
                movement_variability = np.std(acc_array)
                
                features.extend([activity_intensity, estimated_steps, movement_variability])
                feature_names.extend(["activity_intensity", "estimated_steps", "movement_variability"])
            else:
                features.extend([0, 0, 0])
                feature_names.extend(["activity_intensity", "estimated_steps", "movement_variability"])
        
        return features, feature_names
    
    def _normalize_heart_rate(self, hr: float) -> float:
        """Normalize heart rate to 0-1 scale"""
        # Typical range: 50-200 bpm
        return max(0, min(1, (hr - 50) / 150))
    
    def _normalize_spo2(self, spo2: float) -> float:
        """Normalize SpO2 to 0-1 scale"""
        # Typical range: 90-100%
        return max(0, min(1, (spo2 - 90) / 10))
    
    def _categorize_activity(self, acc_magnitude: float) -> float:
        """Categorize activity level based on accelerometer magnitude"""
        if acc_magnitude < 1.1:
            return 0.0  # Sedentary
        elif acc_magnitude < 1.5:
            return 0.5  # Light activity
        else:
            return 1.0  # Moderate/vigorous activity
    
    def _detect_anomaly(self, hr: float, spo2: float, acc: float) -> int:
        """Simple rule-based anomaly detection"""
        anomaly_score = 0
        
        if hr is not None:
            if hr < self.anomaly_thresholds["heart_rate_low"] or hr > self.anomaly_thresholds["heart_rate_high"]:
                anomaly_score += 1
        
        if spo2 is not None:
            if spo2 < self.anomaly_thresholds["spo2_low"]:
                anomaly_score += 1
        
        # If multiple anomalies detected, classify as anomaly
        return 1 if anomaly_score >= 1 else 0
    
    def _detect_windowed_anomaly(self, window_data: Dict[str, List[float]]) -> int:
        """Detect anomalies in windowed data"""
        anomaly_indicators = 0
        
        # Check for sustained anomalies
        if "heart_rate" in window_data:
            hr_values = [v for v in window_data["heart_rate"] if not np.isnan(v)]
            if hr_values:
                hr_array = np.array(hr_values)
                abnormal_hr_ratio = np.mean(
                    (hr_array < self.anomaly_thresholds["heart_rate_low"]) |
                    (hr_array > self.anomaly_thresholds["heart_rate_high"])
                )
                if abnormal_hr_ratio > 0.3:  # 30% of readings abnormal
                    anomaly_indicators += 1
        
        if "spo2" in window_data:
            spo2_values = [v for v in window_data["spo2"] if not np.isnan(v)]
            if spo2_values:
                spo2_array = np.array(spo2_values)
                low_spo2_ratio = np.mean(spo2_array < self.anomaly_thresholds["spo2_low"])
                if low_spo2_ratio > 0.2:  # 20% of readings low
                    anomaly_indicators += 1
        
        return 1 if anomaly_indicators >= 1 else 0
    
    def _calculate_confidence(self, features: List[float]) -> float:
        """Calculate confidence score based on feature quality"""
        if not features:
            return 0.0
        
        # Simple confidence based on feature completeness and variability
        non_zero_features = sum(1 for f in features if abs(f) > 1e-6)
        completeness = non_zero_features / len(features)
        
        # Penalize extreme values (potential outliers)
        feature_array = np.array(features)
        extreme_ratio = np.mean(np.abs(feature_array) > 3 * np.std(feature_array))
        
        confidence = completeness * (1 - extreme_ratio * 0.5)
        return max(0.1, min(1.0, confidence))
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores (placeholder for actual implementation)"""
        # In practice, this would be learned from training data
        importance_map = {
            "heart_rate": 0.25,
            "spo2": 0.20,
            "hrv_rmssd": 0.15,
            "activity_intensity": 0.10,
            "spectral_entropy": 0.08,
            "variability": 0.07,
            "trend": 0.05,
            "other": 0.10
        }
        return importance_map
