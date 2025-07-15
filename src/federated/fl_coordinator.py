"""
Federated Learning Coordinator
Main orchestration component for the Edge-Aware FL system
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import uuid

from ..utils.logger import setup_logger, PerformanceLogger
from ..edge.edge_server import EdgeServer
from ..wearable.device_simulator import WearableDevice
from ..network.network_simulator import NetworkSimulator
from ..evaluation.metrics_collector import MetricsCollector
from config.config import Config


@dataclass
class ExperimentConfig:
    """Configuration for FL experiment"""
    num_rounds: int
    num_edge_servers: int
    num_devices: int
    experiment_duration_hours: float
    data_distribution: str  # "iid", "non_iid"
    device_failure_rate: float
    network_conditions: str  # "stable", "dynamic", "poor"


@dataclass
class ExperimentResults:
    """Results from FL experiment"""
    experiment_id: str
    config: ExperimentConfig
    start_time: datetime
    end_time: datetime
    final_accuracy: float
    convergence_round: Optional[int]
    total_communication_cost: float
    total_energy_consumed: float
    average_round_time: float
    device_participation_rate: float
    edge_server_utilization: Dict[str, float]


class FederatedLearningCoordinator:
    """
    Main coordinator for the Edge-Aware Federated Learning system
    Orchestrates edge servers, devices, and evaluation
    """
    
    def __init__(self, config: Config, network_simulator: NetworkSimulator = None, metrics_collector: MetricsCollector = None):
        self.config = config
        self.logger = setup_logger("FLCoordinator")
        self.perf_logger = PerformanceLogger()
        
        # System components
        self.edge_servers: Dict[str, EdgeServer] = {}
        self.wearable_devices: Dict[str, WearableDevice] = {}
        self.network_simulator = network_simulator or NetworkSimulator(config.NETWORK)
        self.metrics_collector = metrics_collector or MetricsCollector("results")
        
        # Experiment state
        self.is_running = False
        self.current_experiment_id = None
        self.experiment_results: List[ExperimentResults] = []
        self.round_metrics: List[Dict[str, Any]] = []  # Track metrics for each round
        
        # Global FL state
        self.global_round = 0
        self.target_rounds = config.FL.NUM_ROUNDS
        self.convergence_threshold = 0.95
        self.convergence_patience = 10
        
        self.logger.info("Federated Learning Coordinator initialized")
    
    async def setup_experiment(self, experiment_config: ExperimentConfig) -> str:
        """
        Set up a new federated learning experiment
        
        Args:
            experiment_config: Experiment configuration
            
        Returns:
            Experiment ID
        """
        experiment_id = str(uuid.uuid4())
        self.current_experiment_id = experiment_id
        
        try:
            self.logger.info(f"Setting up experiment {experiment_id}")
            
            # Initialize edge servers
            await self._setup_edge_servers(experiment_config.num_edge_servers)
            
            # Initialize wearable devices
            await self._setup_wearable_devices(experiment_config.num_devices)
            
            # Configure network conditions
            await self._configure_network_conditions(experiment_config.network_conditions)
            
            # Initialize metrics collection
            self.metrics_collector.start_experiment(experiment_id, experiment_config)
            
            self.logger.info(
                f"Experiment {experiment_id} setup complete - "
                f"Servers: {experiment_config.num_edge_servers}, "
                f"Devices: {experiment_config.num_devices}"
            )
            
            return experiment_id
            
        except Exception as e:
            self.logger.error(f"Error setting up experiment {experiment_id}: {e}")
            raise
    
    async def _setup_edge_servers(self, num_servers: int):
        """Initialize edge servers"""
        self.logger.info(f"Initializing {num_servers} edge servers")
        
        # Clear any default nodes from NetworkSimulator to avoid conflicts
        self.network_simulator.clear_default_nodes()
        
        # Predefined server locations (major cities)
        server_locations = [
            (40.7128, -74.0060),  # NYC
            (34.0522, -118.2437), # LA
            (41.8781, -87.6298),  # Chicago
            (29.7604, -95.3698),  # Houston
            (33.4484, -112.0740), # Phoenix
        ]
        
        for i in range(num_servers):
            server_id = f"edge_server_{i+1}"
            location = server_locations[i % len(server_locations)]
            
            # Create server config in the format expected by EdgeServer
            server_config = {
                "max_clients": self.config.EDGE.MAX_CLIENTS,
                "min_clients": self.config.EDGE.MIN_CLIENTS,
                "max_wait_time": self.config.EDGE.MAX_WAIT_TIME,
                "aggregation_algorithm": self.config.EDGE.AGGREGATION_ALGORITHM,
                "model": {
                    "architecture": self.config.FL.MODEL_TYPE,
                    "input_features": self.config.FL.INPUT_FEATURES,
                    "num_classes": 2,  # health classification
                    "hidden_units": 64,
                    "dropout_rate": 0.3
                }
            }
            
            # Create server with configuration
            server = EdgeServer(
                server_id=server_id,
                config=server_config,
                location=location,
                network_simulator=self.network_simulator  # Pass shared network simulator
            )
            
            self.edge_servers[server_id] = server
            
            # Register server with network simulator for device discovery, passing the server instance
            self.network_simulator.register_edge_server(
                server_id, 
                location, 
                server_config["max_clients"], 
                server_instance=server
            )
            
            # Start server in background
            asyncio.create_task(server.start_server())
            
            self.logger.info(f"Edge server {server_id} initialized at {location}")
    
    async def _setup_wearable_devices(self, num_devices: int):
        """Initialize wearable devices"""
        self.logger.info(f"Initializing {num_devices} wearable devices")
        
        for i in range(num_devices):
            device_id = f"device_{i+1}"
            
            # Assign random location around edge servers
            base_location = list(self.edge_servers.values())[i % len(self.edge_servers)].location
            
            # Add random offset (within 1km for testing)
            import random
            offset_lat = random.uniform(-0.01, 0.01)  # ~1km
            offset_lon = random.uniform(-0.01, 0.01)
            device_location = (
                base_location[0] + offset_lat,
                base_location[1] + offset_lon
            )
            
            # Create device config in the format expected by WearableDevice
            device_config = {
                "sensors": {},
                "hardware": {
                    "battery_capacity_mah": self.config.DEVICES.BATTERY_CAPACITY
                },
                "training": {
                    "local_epochs": self.config.FL.LOCAL_EPOCHS,
                    "batch_size": self.config.FL.BATCH_SIZE,
                    "learning_rate": self.config.FL.LEARNING_RATE
                },
                "model": {
                    "architecture": self.config.FL.MODEL_TYPE,
                    "input_features": self.config.FL.INPUT_FEATURES,
                    "num_classes": 2,  # health classification
                    "hidden_units": 64,
                    "dropout_rate": 0.3
                }
            }
            
            # Create device
            device = WearableDevice(
                device_id=device_id,
                config=device_config,
                initial_location=device_location,
                network_simulator=self.network_simulator  # Pass shared network simulator
            )
            
            self.wearable_devices[device_id] = device
            
            # Start device monitoring in background
            asyncio.create_task(device.start_monitoring())
            
        self.logger.info(f"Initialized {num_devices} wearable devices")
    
    async def _configure_network_conditions(self, condition_type: str):
        """Configure network simulation conditions"""
        self.logger.info(f"Configuring network conditions: {condition_type}")
        
        # Network configuration based on condition type
        network_configs = {
            "stable": {
                "bandwidth_variance": 0.1,
                "latency_variance": 0.1,
                "packet_loss_rate": 0.001
            },
            "dynamic": {
                "bandwidth_variance": 0.3,
                "latency_variance": 0.3,
                "packet_loss_rate": 0.01
            },
            "poor": {
                "bandwidth_variance": 0.5,
                "latency_variance": 0.5,
                "packet_loss_rate": 0.05
            }
        }
        
        config = network_configs.get(condition_type, network_configs["stable"])
        # Apply configuration to network simulator
        # (Implementation depends on NetworkSimulator capabilities)
    
    async def run_experiment(self, experiment_config: ExperimentConfig) -> ExperimentResults:
        """
        Run a complete federated learning experiment
        
        Args:
            experiment_config: Experiment configuration
            
        Returns:
            Experiment results
        """
        start_time = datetime.now()
        
        try:
            # Reset round metrics for new experiment
            self.round_metrics = []
            
            # Setup experiment
            experiment_id = await self.setup_experiment(experiment_config)
            
            # Wait for initial setup to complete
            self.logger.info("Waiting for devices to connect and collect initial data...")
            await asyncio.sleep(30)  # Increased from 5 to 30 seconds
            
            # Debug: Check device and server status
            self.logger.info(f"Setup complete. Checking status:")
            for server_id, server in self.edge_servers.items():
                self.logger.info(f"Edge server {server_id}: running={getattr(server, 'is_running', False)}, "
                               f"connected_clients={len(getattr(server, 'connected_clients', []))}")
            
            for device_id, device in self.wearable_devices.items():
                status = device.get_device_status()
                self.logger.info(f"Device {device_id}: connected_to={status.connected_edge_node}, "
                               f"data_buffer_size={len(device.local_data_buffer)}, "
                               f"battery={status.battery_level:.1f}%")
            
            self.is_running = True
            self.global_round = 0
            
            self.logger.info(f"Starting experiment {experiment_id}")
            
            # Run federated learning rounds
            convergence_round = None
            final_accuracy = 0.0
            
            while (self.global_round < experiment_config.num_rounds and 
                   self.is_running):
                
                round_start_time = datetime.now()
                
                # Execute one round of federated learning
                round_results = await self._execute_fl_round()
                
                if round_results:
                    final_accuracy = round_results.get("global_accuracy", 0.0)
                    
                    # Store round metrics for visualization
                    round_metrics = {
                        'round': self.global_round,
                        'global_accuracy': final_accuracy,
                        'global_loss': round_results.get("global_loss", 0.0),
                        'communication_cost': round_results.get("communication_cost", 0.0),
                        'energy_consumed': round_results.get("energy_consumed", 0.0),
                        'participating_servers': round_results.get("participating_servers", [])
                    }
                    self.round_metrics.append(round_metrics)
                    
                    # Check for convergence
                    if (final_accuracy >= self.convergence_threshold and 
                        convergence_round is None):
                        convergence_round = self.global_round
                
                # Log round completion
                round_duration = (datetime.now() - round_start_time).total_seconds()
                self.logger.info(
                    f"Round {self.global_round} completed - "
                    f"Accuracy: {final_accuracy:.4f}, "
                    f"Duration: {round_duration:.2f}s"
                )
                
                self.global_round += 1
                
                # Brief pause between rounds
                await asyncio.sleep(2)
            
            end_time = datetime.now()
            
            # Collect final results
            results = await self._collect_experiment_results(
                experiment_id, experiment_config, start_time, end_time,
                final_accuracy, convergence_round
            )
            
            self.experiment_results.append(results)
            
            # Save results to JSON file for visualization
            self.save_results(results)
            
            self.logger.info(
                f"Experiment {experiment_id} completed - "
                f"Final accuracy: {final_accuracy:.4f}, "
                f"Convergence round: {convergence_round or 'N/A'}"
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error running experiment: {e}")
            raise
        finally:
            self.is_running = False
            await self._cleanup_experiment()
    
    async def _execute_fl_round(self) -> Optional[Dict[str, Any]]:
        """Execute a single federated learning round"""
        try:
            round_results = {}
            participating_servers = []
            
            # Collect results from all active edge servers
            for server_id, server in self.edge_servers.items():
                if server.is_running and len(server.connected_clients) >= server.min_clients_for_aggregation:
                    participating_servers.append(server_id)
            
            if not participating_servers:
                self.logger.warning(f"No edge servers ready for round {self.global_round}")
                # Debug: Check server status
                for server_id, server in self.edge_servers.items():
                    self.logger.info(f"Server {server_id}: running={getattr(server, 'is_running', False)}, "
                                   f"clients={len(getattr(server, 'connected_clients', []))}, "
                                   f"min_clients={getattr(server, 'min_clients_for_aggregation', 0)}")
                return None
            
            # Wait for edge servers to complete their local rounds
            # (Edge servers run their own aggregation loops)
            self.logger.info(f"Round {self.global_round}: Waiting for training and aggregation...")
            await asyncio.sleep(60)  # Increased from 10 to 60 seconds for training
            
            # Collect metrics from this round
            round_metrics = await self._collect_round_metrics(participating_servers)
            
            # Calculate global metrics (simplified)
            if round_metrics:
                avg_accuracy = sum(
                    metrics.get("accuracy", 0.0) 
                    for metrics in round_metrics.values()
                ) / len(round_metrics)
                
                round_results["global_accuracy"] = avg_accuracy
                round_results["participating_servers"] = participating_servers
                round_results["round_metrics"] = round_metrics
            
            return round_results
            
        except Exception as e:
            self.logger.error(f"Error executing FL round {self.global_round}: {e}")
            return None
    
    async def _collect_round_metrics(
        self, 
        participating_servers: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Collect metrics from participating edge servers"""
        round_metrics = {}
        
        for server_id in participating_servers:
            server = self.edge_servers[server_id]
            
            # Get server status and metrics
            server_status = server.get_server_status()
            round_metrics[server_id] = {
                "connected_clients": server_status["connected_clients"],
                "current_round": server_status["current_round"],
                "accuracy": server_status["performance_metrics"].get("model_accuracy_history", [0])[-1] if server_status["performance_metrics"].get("model_accuracy_history") else 0,
                "communication_cost": server_status["performance_metrics"].get("communication_overhead", 0),
                "round_time": server_status["performance_metrics"].get("average_round_time", 0)
            }
        
        return round_metrics
    
    async def _collect_experiment_results(
        self,
        experiment_id: str,
        config: ExperimentConfig,
        start_time: datetime,
        end_time: datetime,
        final_accuracy: float,
        convergence_round: Optional[int]
    ) -> ExperimentResults:
        """Collect comprehensive experiment results"""
        
        # Aggregate metrics from all components
        total_communication_cost = 0.0
        total_energy_consumed = 0.0
        total_round_time = 0.0
        device_participation_counts = {}
        server_utilizations = {}
        
        # Collect from edge servers
        for server_id, server in self.edge_servers.items():
            server_status = server.get_server_status()
            metrics = server_status["performance_metrics"]
            
            total_communication_cost += metrics.get("communication_overhead", 0)
            total_round_time += metrics.get("average_round_time", 0)
            server_utilizations[server_id] = server_status["load_percentage"]
        
        # Collect from devices
        for device_id, device in self.wearable_devices.items():
            device_metrics = device.get_performance_metrics()
            total_energy_consumed += device_metrics.get("total_energy_consumption", 0)
            device_participation_counts[device_id] = device_metrics.get("training_rounds", 0)
        
        # Calculate averages
        num_servers = len(self.edge_servers)
        num_devices = len(self.wearable_devices)
        
        avg_round_time = total_round_time / max(1, num_servers)
        
        # Calculate participation rate
        total_possible_participations = config.num_rounds * num_devices
        actual_participations = sum(device_participation_counts.values())
        participation_rate = actual_participations / max(1, total_possible_participations)
        
        return ExperimentResults(
            experiment_id=experiment_id,
            config=config,
            start_time=start_time,
            end_time=end_time,
            final_accuracy=final_accuracy,
            convergence_round=convergence_round,
            total_communication_cost=total_communication_cost,
            total_energy_consumed=total_energy_consumed,
            average_round_time=avg_round_time,
            device_participation_rate=participation_rate,
            edge_server_utilization=server_utilizations
        )
    
    async def _cleanup_experiment(self):
        """Clean up experiment resources"""
        self.logger.info("Cleaning up experiment resources")
        
        # Stop all edge servers
        cleanup_tasks = []
        for server_id, server in self.edge_servers.items():
            cleanup_tasks.append(server.stop_server())
            # Unregister from network simulator
            self.network_simulator.unregister_edge_server(server_id)
        
        # Stop all devices (devices handle their own cleanup)
        for device in self.wearable_devices.values():
            device.is_training = False  # Signal to stop
        
        # Wait for cleanup
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        # Clear component dictionaries
        self.edge_servers.clear()
        self.wearable_devices.clear()
        
        self.logger.info("Experiment cleanup completed")

    def save_results(self, results: ExperimentResults, results_dir: str = "results"):
        """Save experiment results to JSON file"""
        import os
        from datetime import datetime
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Create experiment subdirectory
        experiment_dir = os.path.join(results_dir, results.experiment_id)
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Convert results to dictionary for JSON serialization
        results_dict = asdict(results)
        
        # Convert datetime objects to strings
        results_dict['start_time'] = results.start_time.isoformat()
        results_dict['end_time'] = results.end_time.isoformat()
        results_dict['config'] = asdict(results.config)
        
        # Add round-by-round data from metrics collector if available
        if hasattr(self, 'round_metrics'):
            results_dict['rounds'] = list(range(1, len(self.round_metrics) + 1))
            results_dict['accuracy'] = [m.get('global_accuracy', 0) for m in self.round_metrics]
            results_dict['loss'] = [m.get('global_loss', 0) for m in self.round_metrics]
            results_dict['communication_overhead'] = [m.get('communication_cost', 0) for m in self.round_metrics]
            results_dict['energy_consumption'] = [m.get('energy_consumed', 0) for m in self.round_metrics]
        
        # Save to JSON file
        results_file = os.path.join(experiment_dir, 'results.json')
        try:
            with open(results_file, 'w') as f:
                json.dump(results_dict, f, indent=2)
            self.logger.info(f"Results saved to {results_file}")
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
    
    async def run_comparative_study(
        self, 
        base_config: ExperimentConfig,
        variations: List[Dict[str, Any]]
    ) -> List[ExperimentResults]:
        """
        Run a comparative study with multiple experiment variations
        
        Args:
            base_config: Base experiment configuration
            variations: List of configuration variations to test
            
        Returns:
            List of experiment results for comparison
        """
        self.logger.info(f"Starting comparative study with {len(variations)} variations")
        
        comparative_results = []
        
        for i, variation in enumerate(variations):
            self.logger.info(f"Running variation {i+1}/{len(variations)}: {variation}")
            
            # Create modified config
            modified_config = ExperimentConfig(**{**asdict(base_config), **variation})
            
            try:
                # Run experiment
                results = await self.run_experiment(modified_config)
                comparative_results.append(results)
                
                # Brief pause between experiments
                await asyncio.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Error in variation {i+1}: {e}")
                continue
        
        self.logger.info(f"Comparative study completed - {len(comparative_results)} successful runs")
        return comparative_results
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of all completed experiments"""
        if not self.experiment_results:
            return {"message": "No completed experiments"}
        
        # Aggregate statistics
        accuracies = [result.final_accuracy for result in self.experiment_results]
        round_times = [result.average_round_time for result in self.experiment_results]
        communication_costs = [result.total_communication_cost for result in self.experiment_results]
        energy_consumptions = [result.total_energy_consumed for result in self.experiment_results]
        
        return {
            "total_experiments": len(self.experiment_results),
            "average_final_accuracy": sum(accuracies) / len(accuracies),
            "best_accuracy": max(accuracies),
            "worst_accuracy": min(accuracies),
            "average_round_time": sum(round_times) / len(round_times),
            "total_communication_cost": sum(communication_costs),
            "total_energy_consumed": sum(energy_consumptions),
            "convergence_rate": sum(1 for result in self.experiment_results if result.convergence_round) / len(self.experiment_results),
            "experiments": [asdict(result) for result in self.experiment_results]
        }
    
    async def stop_current_experiment(self):
        """Stop the currently running experiment"""
        if self.is_running:
            self.logger.info("Stopping current experiment")
            self.is_running = False
            await self._cleanup_experiment()
        else:
            self.logger.info("No experiment currently running")

# Alias for convenience
FLCoordinator = FederatedLearningCoordinator
