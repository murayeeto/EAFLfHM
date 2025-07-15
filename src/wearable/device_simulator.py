"""
Wearable Device Simulator
Simulates Android smartwatches with physiological sensors
"""

import asyncio
import json
import logging
import random
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..utils.logger import setup_logger
from ..utils.crypto import encrypt_data, decrypt_data, generate_keypair
from models.health_models import create_model, count_parameters
from ..network.network_simulator import NetworkSimulator
from .sensor_simulator import SensorSimulator
from .battery_manager import BatteryManager
from .data_processor import HealthDataProcessor


@dataclass
class DeviceStatus:
    """Device status information"""
    device_id: str
    battery_level: float
    signal_strength: float
    connected_edge_node: Optional[str]
    is_training: bool
    last_update: datetime
    location: Tuple[float, float]  # (latitude, longitude)


class WearableDevice:
    """
    Simulates an Android smartwatch with health monitoring capabilities
    """
    
    def __init__(
        self,
        device_id: str = None,
        config: Dict[str, Any] = None,
        initial_location: Tuple[float, float] = None,
        network_simulator: NetworkSimulator = None
    ):
        self.device_id = device_id or str(uuid.uuid4())
        self.config = config or {}
        self.logger = setup_logger(f"Device-{self.device_id[:8]}")
        
        # Initialize components
        self.sensor_simulator = SensorSimulator(config.get("sensors", {}))
        self.battery_manager = BatteryManager(
            config.get("hardware", {}).get("battery_capacity_mah", 300)
        )
        self.data_processor = HealthDataProcessor()
        self.network_simulator = network_simulator or NetworkSimulator()
        
        # Device state
        self.location = initial_location or (40.7128, -74.0060)  # Default: NYC
        self.velocity = (0.0, 0.0)  # m/s
        self.connected_edge_node = None
        self.is_training = False
        self.model = None
        self.local_data_buffer = []
        
        # Federated learning state
        self.round_number = 0
        self.local_epochs = config.get("training", {}).get("local_epochs", 5)
        self.batch_size = config.get("training", {}).get("batch_size", 32)
        self.learning_rate = config.get("training", {}).get("learning_rate", 0.001)
        
        # Security
        self.private_key, self.public_key = generate_keypair()
        
        # Performance metrics
        self.metrics = {
            "training_time": [],
            "communication_cost": [],
            "energy_consumption": [],
            "model_accuracy": []
        }
        
        self.logger.info(f"Initialized wearable device {self.device_id}")
    
    async def start_monitoring(self):
        """Start continuous health monitoring"""
        self.logger.info("Starting health monitoring")
        
        # Initialize model
        await self._initialize_model()
        
        # Start monitoring tasks
        monitoring_tasks = [
            self._sensor_data_collection(),
            self._edge_node_discovery(),
            self._model_training_loop(),
            self._battery_monitoring(),
            self._mobility_simulation()
        ]
        
        await asyncio.gather(*monitoring_tasks)
    
    async def _initialize_model(self):
        """Initialize the local neural network model"""
        model_config = self.config.get("model", {})
        
        self.model = create_model(
            architecture=model_config.get("architecture", "1d_cnn"),
            input_features=model_config.get("input_features", 3),
            num_classes=model_config.get("num_classes", 2),
            lstm_hidden=model_config.get("hidden_units", 64),  # Map hidden_units to lstm_hidden
            dropout_rate=model_config.get("dropout_rate", 0.3),
            sequence_length=1  # Single feature vectors, not sequences
        )
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate
        )
        self.criterion = nn.CrossEntropyLoss()
        
        param_count = count_parameters(self.model)
        self.logger.info(f"Initialized model with {param_count} parameters")
    
    async def _sensor_data_collection(self):
        """Continuously collect sensor data"""
        while True:
            try:
                # Collect sensor readings
                sensor_data = await self.sensor_simulator.collect_readings()
                
                # Process and store data
                processed_data = self.data_processor.process_realtime_data(sensor_data)
                self.local_data_buffer.append(processed_data)
                
                # Add debugging
                if len(self.local_data_buffer) % 10 == 0:  # Log every 10 samples
                    self.logger.info(f"Data buffer size: {len(self.local_data_buffer)}, Latest data: {processed_data}")
                
                # Limit buffer size (memory management)
                if len(self.local_data_buffer) > 1000:
                    self.local_data_buffer = self.local_data_buffer[-800:]
                
                # Update battery consumption
                power_consumption = sum(
                    sensor["power_consumption_mw"] 
                    for sensor in self.config.get("sensors", {}).values()
                )
                self.battery_manager.consume_power(power_consumption / 1000)  # Convert to watts
                
                await asyncio.sleep(1)  # 1 Hz data collection
                
            except Exception as e:
                self.logger.error(f"Error in sensor data collection: {e}")
                await asyncio.sleep(5)
    
    async def _edge_node_discovery(self):
        """Discover and connect to nearby edge nodes"""
        while True:
            try:
                # Simulate edge node discovery
                available_nodes = await self.network_simulator.discover_edge_nodes(
                    self.location, 
                    signal_threshold=-80
                )
                
                self.logger.info(f"Edge node discovery: found {len(available_nodes) if available_nodes else 0} nodes")
                
                if available_nodes:
                    # Select best edge node based on multiple criteria
                    best_node = await self._select_optimal_edge_node(available_nodes)
                    
                    if best_node != self.connected_edge_node:
                        await self._connect_to_edge_node(best_node)
                else:
                    self.logger.warning("No edge nodes discovered")
                
                await asyncio.sleep(30)  # Discovery every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in edge node discovery: {e}")
                await asyncio.sleep(10)
    
    async def _select_optimal_edge_node(
        self, 
        available_nodes: List[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Select optimal edge node based on multiple criteria
        """
        if not available_nodes:
            return None
        
        best_node = None
        best_score = -1
        
        for node in available_nodes:
            # Calculate selection score based on:
            # - Signal strength (40%)
            # - Server load (30%)
            # - Distance (20%)
            # - Battery level consideration (10%)
            
            signal_score = (node["signal_strength"] + 100) / 50  # Normalize to 0-1
            load_score = 1 - (node["load"] / 100)  # Lower load is better
            distance_score = 1 / (1 + node["distance"] / 1000)  # Closer is better
            battery_score = self.battery_manager.get_battery_level() / 100
            
            total_score = (
                0.4 * signal_score +
                0.3 * load_score +
                0.2 * distance_score +
                0.1 * battery_score
            )
            
            if total_score > best_score:
                best_score = total_score
                best_node = node["node_id"]
        
        return best_node
    
    async def _connect_to_edge_node(self, node_id: str):
        """Connect to a specific edge node"""
        try:
            connection_result = await self.network_simulator.connect_to_edge_node(
                self.device_id, 
                node_id, 
                self.public_key
            )
            
            if connection_result["success"]:
                self.connected_edge_node = node_id
                self.logger.info(f"Connected to edge node {node_id}")
            else:
                self.logger.warning(f"Failed to connect to edge node {node_id}")
                
        except Exception as e:
            self.logger.error(f"Error connecting to edge node {node_id}: {e}")
    
    async def _model_training_loop(self):
        """Local model training loop"""
        while True:
            try:
                buffer_size = len(self.local_data_buffer)
                self.logger.info(f"Training check: buffer_size={buffer_size}, batch_size={self.batch_size}, connected={self.connected_edge_node}, is_training={self.is_training}")
                
                if (buffer_size >= self.batch_size and 
                    self.connected_edge_node and 
                    not self.is_training):
                    
                    self.logger.info("Starting local training...")
                    await self._perform_local_training()
                else:
                    reasons = []
                    if buffer_size < self.batch_size:
                        reasons.append(f"insufficient data ({buffer_size}/{self.batch_size})")
                    if not self.connected_edge_node:
                        reasons.append("not connected to edge server")
                    if self.is_training:
                        reasons.append("already training")
                    self.logger.info(f"Training skipped: {', '.join(reasons)}")
                
                await asyncio.sleep(60)  # Check for training every minute
                
            except Exception as e:
                self.logger.error(f"Error in model training loop: {e}")
                await asyncio.sleep(30)
    
    async def _perform_local_training(self):
        """Perform local federated learning training"""
        self.is_training = True
        start_time = time.time()
        
        try:
            # Prepare training data
            train_loader = self._prepare_training_data()
            
            # Local training
            self.model.train()
            total_loss = 0
            
            for epoch in range(self.local_epochs):
                for batch_data, batch_labels in train_loader:
                    self.optimizer.zero_grad()
                    outputs = self.model(batch_data)
                    loss = self.criterion(outputs, batch_labels)
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()
            
            training_time = time.time() - start_time
            avg_loss = total_loss / (len(train_loader) * self.local_epochs)
            
            # Update battery consumption for training
            training_power = 2.0  # Watts (higher during training)
            self.battery_manager.consume_power(training_power * training_time / 3600)
            
            # Send model updates to edge server
            await self._send_model_updates()
            
            # Record metrics
            self.metrics["training_time"].append(training_time)
            self.metrics["energy_consumption"].append(training_power * training_time)
            
            self.logger.info(
                f"Local training completed - Loss: {avg_loss:.4f}, "
                f"Time: {training_time:.2f}s"
            )
            
        except Exception as e:
            self.logger.error(f"Error during local training: {e}")
        finally:
            self.is_training = False
    
    def _prepare_training_data(self) -> DataLoader:
        """Prepare training data from local buffer"""
        # Convert buffer to tensors
        recent_data = self.local_data_buffer[-200:]  # Use recent data
        
        # Extract features and labels
        features = []
        labels = []
        
        for data_point in recent_data:
            # Handle ProcessedHealthData objects
            if hasattr(data_point, 'features') and hasattr(data_point, 'label'):
                if data_point.features and data_point.label is not None:
                    features.append(data_point.features)
                    labels.append(data_point.label)
        
        if len(features) < self.batch_size:
            # Generate synthetic data if insufficient real data
            features, labels = self._generate_synthetic_training_data()
        
        # Convert to tensors
        X = torch.FloatTensor(features)
        y = torch.LongTensor(labels)
        
        # Reshape for CNN (batch_size, channels, sequence_length)
        # For health data: features become channels, sequence_length = 1
        if len(X.shape) == 2:
            X = X.unsqueeze(2)  # Add sequence dimension: [batch, features] -> [batch, features, 1]
        
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
    
    def _generate_synthetic_training_data(self) -> Tuple[List, List]:
        """Generate synthetic training data for demonstration"""
        features = []
        labels = []
        
        for _ in range(self.batch_size * 2):
            # Generate synthetic physiological data to match ProcessedHealthData format
            hr = np.random.normal(75, 15)  # Heart rate
            spo2 = np.random.normal(98, 2)  # SpO2
            activity = np.random.normal(0.5, 0.3)  # Activity level
            
            # Create 8-feature vector to match data processor output
            feature_vector = [
                hr,                              # heart_rate
                (hr - 60) / 40,                 # heart_rate_normalized  
                spo2,                           # spo2
                (spo2 - 95) / 5,               # spo2_normalized
                np.random.normal(1.0, 0.3),     # accelerometer_magnitude
                max(0, activity),               # activity_level
                np.sin(2 * np.pi * 12 / 24),   # hour_sin (noon)
                np.cos(2 * np.pi * 12 / 24)    # hour_cos (noon)
            ]
            
            # Simple rule-based labeling (anomaly detection)
            is_anomaly = (hr < 50 or hr > 120 or spo2 < 95)
            label = 1 if is_anomaly else 0
            
            features.append(feature_vector)
            labels.append(label)
        
        return features, labels
    
    async def _send_model_updates(self):
        """Send encrypted model updates to edge server"""
        try:
            # Get model parameters
            model_state = self.model.state_dict()
            
            # Serialize and encrypt
            model_data = {
                "device_id": self.device_id,
                "round_number": self.round_number,
                "model_state": model_state,
                "data_size": len(self.local_data_buffer),
                "timestamp": datetime.now().isoformat()
            }
            
            serialized_data = json.dumps(
                model_data, 
                default=lambda x: x.tolist() if isinstance(x, torch.Tensor) else str(x)
            )
            
            # Temporarily disable encryption for development/testing
            encrypted_data = serialized_data.encode('utf-8')
            # TODO: Re-enable encryption: encrypted_data = encrypt_data(serialized_data, self.public_key)
            
            # Send to edge server
            success = await self.network_simulator.send_model_update(
                self.device_id,
                self.connected_edge_node,
                encrypted_data
            )
            
            if success:
                self.round_number += 1
                self.logger.info("Model updates sent successfully")
                
                # Record communication cost
                data_size_mb = len(encrypted_data) / (1024 * 1024)
                self.metrics["communication_cost"].append(data_size_mb)
            else:
                self.logger.warning("Failed to send model updates")
                
        except Exception as e:
            self.logger.error(f"Error sending model updates: {e}")
    
    async def receive_global_model(self, encrypted_model_data: bytes):
        """Receive and apply global model updates from edge server"""
        try:
            # Decrypt model data
            decrypted_data = decrypt_data(encrypted_model_data, self.private_key)
            model_update = json.loads(decrypted_data)
            
            # Update local model
            global_state = model_update["global_model_state"]
            
            # Convert back to tensors
            for key, value in global_state.items():
                if isinstance(value, list):
                    global_state[key] = torch.FloatTensor(value)
            
            self.model.load_state_dict(global_state)
            
            self.logger.info("Global model update received and applied")
            
        except Exception as e:
            self.logger.error(f"Error receiving global model: {e}")
    
    async def _battery_monitoring(self):
        """Monitor battery level and adjust behavior"""
        while True:
            try:
                battery_level = self.battery_manager.get_battery_level()
                
                if battery_level < 20:
                    # Enter power saving mode
                    self.logger.warning(f"Low battery: {battery_level:.1f}%")
                    # Reduce sensor sampling rate, training frequency, etc.
                    
                elif battery_level < 5:
                    # Emergency mode
                    self.logger.critical("Critical battery level - entering emergency mode")
                    # Minimal operation only
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in battery monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _mobility_simulation(self):
        """Simulate device mobility patterns"""
        while True:
            try:
                # Simple random walk mobility model
                dx = random.gauss(0, 0.001)  # Small random movement
                dy = random.gauss(0, 0.001)
                
                self.location = (
                    self.location[0] + dx,
                    self.location[1] + dy
                )
                
                # Update velocity
                self.velocity = (dx * 111000, dy * 111000)  # Convert to m/s
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in mobility simulation: {e}")
                await asyncio.sleep(30)
    
    def get_device_status(self) -> DeviceStatus:
        """Get current device status"""
        signal_strength = -50 if self.connected_edge_node else -100
        
        return DeviceStatus(
            device_id=self.device_id,
            battery_level=self.battery_manager.get_battery_level(),
            signal_strength=signal_strength,
            connected_edge_node=self.connected_edge_node,
            is_training=self.is_training,
            last_update=datetime.now(),
            location=self.location
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            "device_id": self.device_id,
            "avg_training_time": np.mean(self.metrics["training_time"]) if self.metrics["training_time"] else 0,
            "total_communication_cost": sum(self.metrics["communication_cost"]),
            "total_energy_consumption": sum(self.metrics["energy_consumption"]),
            "training_rounds": self.round_number,
            "data_points_collected": len(self.local_data_buffer)
        }
