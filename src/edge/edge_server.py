"""
Edge Server Implementation
Manages federated learning aggregation and client coordination
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn
from collections import defaultdict
import numpy as np

from ..utils.logger import setup_logger, PerformanceLogger
from ..utils.crypto import decrypt_data, encrypt_data, verify_signature
from models.health_models import create_model, count_parameters
from ..network.network_simulator import NetworkSimulator
from .aggregation_algorithms import FedAvgAggregator, AdaptiveAggregator
from .client_manager import EdgeClientManager


@dataclass
class ClientUpdate:
    """Client model update information"""
    client_id: str
    round_number: int
    model_state: Dict[str, torch.Tensor]
    data_size: int
    training_time: float
    timestamp: datetime
    accuracy: Optional[float] = None
    loss: Optional[float] = None


@dataclass
class AggregationRound:
    """Federated learning round information"""
    round_number: int
    start_time: datetime
    end_time: Optional[datetime] = None
    participating_clients: List[str] = None
    global_model_accuracy: Optional[float] = None
    convergence_metrics: Dict[str, float] = None


class EdgeServer:
    """
    Edge computing server for federated learning
    Manages client connections, model aggregation, and coordination
    """
    
    def __init__(
        self, 
        server_id: str,
        config: Dict[str, Any] = None,
        location: tuple = None,
        network_simulator = None
    ):
        self.server_id = server_id
        self.config = config or {}
        self.location = location or (40.7128, -74.0060)  # Default NYC
        
        # Logging
        self.logger = setup_logger(f"EdgeServer-{server_id}")
        self.perf_logger = PerformanceLogger()
        
        # Server configuration
        self.max_clients = self.config.get("max_clients", 50)
        self.min_clients_for_aggregation = self.config.get("min_clients", 3)
        self.aggregation_timeout = self.config.get("max_wait_time", 120)  # seconds
        self.aggregation_algorithm = self.config.get("aggregation_algorithm", "fedavg")
        
        # Federated learning state
        self.global_model = None
        self.current_round = 0
        self.client_updates: Dict[int, List[ClientUpdate]] = defaultdict(list)
        self.aggregation_history: List[AggregationRound] = []
        
        # Client management
        self.client_manager = EdgeClientManager(self.max_clients)
        self.connected_clients: Set[str] = set()
        self.client_states: Dict[str, Dict[str, Any]] = {}
        
        # Network simulation
        self.network_simulator = network_simulator or NetworkSimulator()
        
        # Aggregation components
        self.aggregator = self._initialize_aggregator()
        
        # Performance metrics
        self.metrics = {
            "total_rounds": 0,
            "total_clients_served": 0,
            "average_round_time": 0.0,
            "model_accuracy_history": [],
            "communication_overhead": 0.0
        }
        
        # Server status
        self.is_running = False
        self.current_load = 0
        self.last_heartbeat = datetime.now()
        
        self.logger.info(f"Edge server {server_id} initialized at {location}")
    
    def _initialize_aggregator(self):
        """Initialize the model aggregation algorithm"""
        if self.aggregation_algorithm == "fedavg":
            return FedAvgAggregator()
        elif self.aggregation_algorithm == "adaptive":
            return AdaptiveAggregator()
        else:
            self.logger.warning(f"Unknown aggregation algorithm: {self.aggregation_algorithm}")
            return FedAvgAggregator()
    
    async def start_server(self):
        """Start the edge server"""
        self.is_running = True
        self.logger.info("Starting edge server")
        
        # Initialize global model
        await self._initialize_global_model()
        
        # Start server tasks
        server_tasks = [
            self._heartbeat_loop(),
            self._client_management_loop(),
            self._aggregation_loop(),
            self._performance_monitoring_loop()
        ]
        
        try:
            await asyncio.gather(*server_tasks)
        except Exception as e:
            self.logger.error(f"Error in server tasks: {e}")
        finally:
            self.is_running = False
    
    async def stop_server(self):
        """Stop the edge server"""
        self.is_running = False
        self.logger.info("Stopping edge server")
        
        # Disconnect all clients
        for client_id in list(self.connected_clients):
            await self.disconnect_client(client_id)
    
    async def _initialize_global_model(self):
        """Initialize the global federated learning model"""
        model_config = self.config.get("model", {})
        
        self.global_model = create_model(
            architecture=model_config.get("architecture", "1d_cnn"),
            input_features=model_config.get("input_features", 3),
            num_classes=model_config.get("num_classes", 2),
            lstm_hidden=model_config.get("hidden_units", 64),  # Map hidden_units to lstm_hidden
            dropout_rate=model_config.get("dropout_rate", 0.3),
            sequence_length=1  # Single feature vectors, not sequences
        )
        
        param_count = count_parameters(self.global_model)
        self.logger.info(f"Global model initialized with {param_count} parameters")
    
    async def _heartbeat_loop(self):
        """Periodic heartbeat and status updates"""
        while self.is_running:
            try:
                self.last_heartbeat = datetime.now()
                self.current_load = len(self.connected_clients)
                
                # Log server status
                self.logger.debug(
                    f"Heartbeat - Clients: {self.current_load}/{self.max_clients}, "
                    f"Round: {self.current_round}"
                )
                
                await asyncio.sleep(30)  # Heartbeat every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(10)
    
    async def _client_management_loop(self):
        """Manage client connections and health checks"""
        while self.is_running:
            try:
                # Check client health
                current_time = datetime.now()
                timeout_threshold = timedelta(minutes=5)
                
                # Identify inactive clients
                inactive_clients = []
                for client_id in list(self.connected_clients):
                    if client_id in self.client_states:
                        last_seen = self.client_states[client_id].get("last_seen")
                        if last_seen and (current_time - last_seen) > timeout_threshold:
                            inactive_clients.append(client_id)
                
                # Disconnect inactive clients
                for client_id in inactive_clients:
                    await self.disconnect_client(client_id, reason="timeout")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in client management loop: {e}")
                await asyncio.sleep(30)
    
    async def _aggregation_loop(self):
        """Main federated learning aggregation loop"""
        while self.is_running:
            try:
                # Check if we have enough clients for aggregation
                if (len(self.connected_clients) >= self.min_clients_for_aggregation and
                    len(self.client_updates[self.current_round]) >= self.min_clients_for_aggregation):
                    
                    await self._perform_aggregation_round()
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in aggregation loop: {e}")
                await asyncio.sleep(30)
    
    async def _perform_aggregation_round(self):
        """Perform a single round of federated aggregation"""
        round_start_time = datetime.now()
        
        try:
            self.logger.info(f"Starting aggregation round {self.current_round}")
            
            # Get client updates for current round
            client_updates = self.client_updates[self.current_round]
            
            if len(client_updates) < self.min_clients_for_aggregation:
                self.logger.warning("Insufficient client updates for aggregation")
                return
            
            # Perform model aggregation
            aggregated_model = await self._aggregate_models(client_updates)
            
            if aggregated_model is not None:
                # Update global model
                self.global_model.load_state_dict(aggregated_model)
                
                # Evaluate global model (if test data available)
                global_accuracy = await self._evaluate_global_model()
                
                # Broadcast updated model to clients
                await self._broadcast_global_model()
                
                # Record round completion
                round_end_time = datetime.now()
                round_duration = (round_end_time - round_start_time).total_seconds()
                
                # Update metrics
                self.metrics["total_rounds"] += 1
                self.metrics["average_round_time"] = (
                    (self.metrics["average_round_time"] * (self.current_round - 1) + round_duration) /
                    self.current_round if self.current_round > 0 else round_duration
                )
                
                if global_accuracy is not None:
                    self.metrics["model_accuracy_history"].append(global_accuracy)
                
                # Store aggregation round info
                aggregation_round = AggregationRound(
                    round_number=self.current_round,
                    start_time=round_start_time,
                    end_time=round_end_time,
                    participating_clients=[update.client_id for update in client_updates],
                    global_model_accuracy=global_accuracy
                )
                self.aggregation_history.append(aggregation_round)
                
                accuracy_str = f"{global_accuracy:.4f}" if global_accuracy is not None else "N/A"
                self.logger.info(
                    f"Aggregation round {self.current_round} completed - "
                    f"Duration: {round_duration:.2f}s, "
                    f"Participants: {len(client_updates)}, "
                    f"Accuracy: {accuracy_str}"
                )
                
                # Move to next round
                self.current_round += 1
                
                # Clean up old round data
                if self.current_round > 10:  # Keep last 10 rounds
                    old_round = self.current_round - 10
                    if old_round in self.client_updates:
                        del self.client_updates[old_round]
            
        except Exception as e:
            self.logger.error(f"Error during aggregation round: {e}")
    
    async def _aggregate_models(self, client_updates: List[ClientUpdate]) -> Optional[Dict[str, torch.Tensor]]:
        """Aggregate client model updates"""
        try:
            # Prepare data for aggregation
            model_states = [update.model_state for update in client_updates]
            data_sizes = [update.data_size for update in client_updates]
            client_ids = [update.client_id for update in client_updates]
            
            # Perform aggregation using selected algorithm
            aggregated_state = self.aggregator.aggregate(
                model_states=model_states,
                weights=data_sizes,
                client_ids=client_ids,
                round_number=self.current_round
            )
            
            return aggregated_state
            
        except Exception as e:
            self.logger.error(f"Error aggregating models: {e}")
            return None
    
    async def _evaluate_global_model(self) -> Optional[float]:
        """Evaluate the global model (placeholder implementation)"""
        try:
            # In practice, this would use a validation dataset
            # For now, return a mock accuracy that improves over time
            base_accuracy = 0.75
            improvement = min(0.2, self.current_round * 0.005)  # Gradual improvement
            noise = np.random.normal(0, 0.02)  # Add some noise
            
            accuracy = base_accuracy + improvement + noise
            return max(0.5, min(0.99, accuracy))
            
        except Exception as e:
            self.logger.error(f"Error evaluating global model: {e}")
            return None
    
    async def _broadcast_global_model(self):
        """Broadcast updated global model to all connected clients"""
        try:
            # Serialize global model
            model_state = self.global_model.state_dict()
            
            model_data = {
                "round_number": self.current_round,
                "model_state": model_state,
                "server_id": self.server_id,
                "timestamp": datetime.now().isoformat()
            }
            
            # Convert tensors to lists for serialization
            serializable_data = {}
            for key, value in model_data.items():
                if key == "model_state":
                    serializable_data[key] = {
                        k: v.tolist() if isinstance(v, torch.Tensor) else v
                        for k, v in value.items()
                    }
                else:
                    serializable_data[key] = value
            
            serialized_data = json.dumps(serializable_data).encode('utf-8')
            
            # Broadcast to connected clients
            broadcast_results = await self.network_simulator.broadcast_global_model(
                node_id=self.server_id,
                encrypted_model_data=serialized_data,  # Should be encrypted in practice
                target_devices=list(self.connected_clients)
            )
            
            # Log broadcast results
            successful_broadcasts = sum(1 for success in broadcast_results.values() if success)
            self.logger.info(
                f"Global model broadcast - Success: {successful_broadcasts}/{len(self.connected_clients)}"
            )
            
            # Update communication metrics
            data_size_mb = len(serialized_data) / (1024 * 1024)
            self.metrics["communication_overhead"] += data_size_mb * len(self.connected_clients)
            
        except Exception as e:
            self.logger.error(f"Error broadcasting global model: {e}")
    
    async def _performance_monitoring_loop(self):
        """Monitor and log performance metrics"""
        while self.is_running:
            try:
                # Log performance metrics periodically
                self.logger.info(
                    f"Performance - Rounds: {self.metrics['total_rounds']}, "
                    f"Avg Round Time: {self.metrics['average_round_time']:.2f}s, "
                    f"Clients Served: {self.metrics['total_clients_served']}, "
                    f"Communication: {self.metrics['communication_overhead']:.2f}MB"
                )
                
                await asyncio.sleep(300)  # Log every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(60)
    
    async def connect_client(self, client_id: str, client_info: Dict[str, Any]) -> bool:
        """
        Connect a new client to the edge server
        
        Args:
            client_id: Unique client identifier
            client_info: Client information and capabilities
            
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Check server capacity
            if len(self.connected_clients) >= self.max_clients:
                self.logger.warning(f"Server at capacity, rejecting client {client_id}")
                return False
            
            # Check if client already connected
            if client_id in self.connected_clients:
                self.logger.warning(f"Client {client_id} already connected")
                return True
            
            # Add client to connected set
            self.connected_clients.add(client_id)
            self.client_states[client_id] = {
                "info": client_info,
                "connected_at": datetime.now(),
                "last_seen": datetime.now(),
                "updates_sent": 0,
                "total_training_time": 0.0
            }
            
            # Register with client manager
            await self.client_manager.register_client(client_id, client_info)
            
            # Update metrics
            self.metrics["total_clients_served"] += 1
            
            self.logger.info(f"Client {client_id} connected - Total clients: {len(self.connected_clients)}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error connecting client {client_id}: {e}")
            return False
    
    async def disconnect_client(self, client_id: str, reason: str = "voluntary"):
        """
        Disconnect a client from the edge server
        
        Args:
            client_id: Client identifier
            reason: Reason for disconnection
        """
        try:
            if client_id in self.connected_clients:
                self.connected_clients.remove(client_id)
                
                # Clean up client state
                if client_id in self.client_states:
                    connection_duration = (
                        datetime.now() - self.client_states[client_id]["connected_at"]
                    ).total_seconds()
                    
                    self.logger.info(
                        f"Client {client_id} disconnected ({reason}) - "
                        f"Duration: {connection_duration:.0f}s"
                    )
                    
                    del self.client_states[client_id]
                
                # Unregister from client manager
                await self.client_manager.unregister_client(client_id)
                
        except Exception as e:
            self.logger.error(f"Error disconnecting client {client_id}: {e}")
    
    async def receive_model_update(
        self, 
        client_id: str, 
        encrypted_update: bytes
    ) -> bool:
        """
        Receive and process model update from client
        
        Args:
            client_id: Source client ID
            encrypted_update: Encrypted model update
            
        Returns:
            True if update processed successfully
        """
        try:
            if client_id not in self.connected_clients:
                self.logger.warning(f"Received update from unconnected client {client_id}")
                return False
            
            # Decrypt update (in practice, would use proper key management)
            # Temporarily disabled encryption for development/testing
            try:
                update_data = json.loads(encrypted_update.decode('utf-8'))
            except Exception as e:
                self.logger.error(f"Failed to parse update data: {e}")
                return False
            
            # Validate update
            if not self._validate_client_update(update_data, client_id):
                self.logger.warning(f"Invalid update from client {client_id}")
                return False
            
            # Convert model state back to tensors
            model_state = {}
            for key, value in update_data["model_state"].items():
                if isinstance(value, list):
                    model_state[key] = torch.FloatTensor(value)
                else:
                    model_state[key] = value
            
            # Create client update object
            client_update = ClientUpdate(
                client_id=client_id,
                round_number=update_data["round_number"],
                model_state=model_state,
                data_size=update_data["data_size"],
                training_time=update_data.get("training_time", 0.0),
                timestamp=datetime.now(),
                accuracy=update_data.get("accuracy"),
                loss=update_data.get("loss")
            )
            
            # Store update for aggregation
            self.client_updates[client_update.round_number].append(client_update)
            
            # Update client state
            if client_id in self.client_states:
                self.client_states[client_id]["last_seen"] = datetime.now()
                self.client_states[client_id]["updates_sent"] += 1
                self.client_states[client_id]["total_training_time"] += client_update.training_time
            
            # Log performance metrics
            self.perf_logger.log_training_metrics(
                device_id=client_id,
                round_num=client_update.round_number,
                training_time=client_update.training_time,
                loss=client_update.loss or 0.0,
                accuracy=client_update.accuracy or 0.0
            )
            
            self.logger.info(
                f"Received update from {client_id} for round {client_update.round_number} "
                f"(data size: {client_update.data_size})"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing update from client {client_id}: {e}")
            return False
    
    def _validate_client_update(self, update_data: Dict[str, Any], client_id: str) -> bool:
        """Validate client model update"""
        required_fields = ["round_number", "model_state", "data_size"]
        
        for field in required_fields:
            if field not in update_data:
                self.logger.warning(f"Missing required field '{field}' in update from {client_id}")
                return False
        
        # Check round number validity
        round_number = update_data["round_number"]
        if round_number < 0 or round_number > self.current_round + 1:
            self.logger.warning(f"Invalid round number {round_number} from {client_id} (server at round {self.current_round})")
            return False
        
        # Check data size
        if update_data["data_size"] <= 0:
            self.logger.warning(f"Invalid data size {update_data['data_size']} from {client_id}")
            return False
        
        self.logger.info(f"Validation passed for {client_id}: round={round_number}, data_size={update_data['data_size']}")
        return True
    
    def get_server_status(self) -> Dict[str, Any]:
        """Get comprehensive server status"""
        return {
            "server_id": self.server_id,
            "location": self.location,
            "is_running": self.is_running,
            "current_round": self.current_round,
            "connected_clients": len(self.connected_clients),
            "max_clients": self.max_clients,
            "load_percentage": (len(self.connected_clients) / self.max_clients) * 100,
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "aggregation_algorithm": self.aggregation_algorithm,
            "performance_metrics": self.metrics,
            "client_list": list(self.connected_clients)
        }
    
    def get_aggregation_history(self, rounds: int = 10) -> List[Dict[str, Any]]:
        """Get recent aggregation history"""
        recent_rounds = self.aggregation_history[-rounds:] if rounds > 0 else self.aggregation_history
        return [asdict(round_info) for round_info in recent_rounds]
