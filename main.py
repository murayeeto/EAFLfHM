#!/usr/bin/env python3
"""
Main entry point for the Edge-Aware Federated Learning System
"""

import argparse
import asyncio
import logging
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.config import Config
from src.federated.fl_coordinator import FLCoordinator
from src.edge.edge_server import EdgeServer
from src.wearable.device_simulator import WearableDevice
from src.network.network_simulator import NetworkSimulator
from src.evaluation.metrics_collector import MetricsCollector
from src.utils.logger import setup_logging

def setup_argument_parser():
    """Setup command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Edge-Aware Federated Learning for Real-Time Health Monitoring"
    )
    
    parser.add_argument(
        "--mode", 
        choices=["full", "edge-server", "client", "coordinator"], 
        default="full",
        help="Mode to run the system in"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/config.py",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--num-clients", 
        type=int, 
        default=None,
        help="Number of wearable clients to simulate"
    )
    
    parser.add_argument(
        "--num-rounds", 
        type=int, 
        default=None,
        help="Number of FL rounds to run"
    )
    
    parser.add_argument(
        "--edge-server-id", 
        type=str, 
        default="edge_1",
        help="ID of the edge server (for edge-server mode)"
    )
    
    parser.add_argument(
        "--client-id", 
        type=str, 
        default="device_1",
        help="ID of the client device (for client mode)"
    )
    
    parser.add_argument(
        "--log-level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="results",
        help="Directory to save results"
    )
    
    return parser

async def run_full_system(config, args):
    """Run the complete federated learning system simulation"""
    logger = logging.getLogger(__name__)
    logger.info("Starting Edge-Aware Federated Learning System")
    
    # Initialize components
    network_sim = NetworkSimulator(config.NETWORK)
    metrics_collector = MetricsCollector(args.output_dir)
    
    # Create FL coordinator
    coordinator = FLCoordinator(config, network_sim, metrics_collector)
    
    # Override config with command line arguments if provided
    num_clients = args.num_clients or config.SIMULATION.NUM_DEVICES
    num_rounds = args.num_rounds or config.FL.NUM_ROUNDS
    
    logger.info(f"Running FL with {num_clients} clients for {num_rounds} rounds")
    
    try:
        # Create experiment configuration
        from src.federated.fl_coordinator import ExperimentConfig
        experiment_config = ExperimentConfig(
            num_rounds=num_rounds,
            num_edge_servers=config.EDGE.get('NUM_SERVERS', 3),
            num_devices=num_clients,
            experiment_duration_hours=config.SIMULATION.get('DURATION_HOURS', 1.0),
            data_distribution=config.DATA.get('DISTRIBUTION', 'iid'),
            device_failure_rate=config.SIMULATION.get('DEVICE_FAILURE_RATE', 0.1),
            network_conditions=config.NETWORK.get('CONDITIONS', 'stable')
        )
        
        # Run the federated learning experiment
        results = await coordinator.run_experiment(experiment_config)
        
        logger.info("Experiment completed successfully")
        logger.info(f"Results saved to: {args.output_dir}")
        
        # Print summary results
        if results:
            print("\\n=== Experiment Summary ===")
            print(f"Experiment ID: {getattr(results, 'experiment_id', 'N/A')}")
            print(f"Final accuracy: {getattr(results, 'final_accuracy', 0.0):.4f}")
            print(f"Convergence round: {getattr(results, 'convergence_round', 'N/A')}")
            print(f"Total communication cost: {getattr(results, 'total_communication_cost', 0.0):.2f} MB")
            print(f"Total energy consumed: {getattr(results, 'total_energy_consumed', 0.0):.2f} J")
            print(f"Average round time: {getattr(results, 'average_round_time', 0.0):.2f}s")
        
    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        raise

async def run_edge_server(config, args):
    """Run a single edge server"""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting edge server: {args.edge_server_id}")
    
    # Create edge server
    edge_server = EdgeServer(
        server_id=args.edge_server_id,
        config=config.EDGE
    )
    
    try:
        # Start the edge server
        await edge_server.start_server()
        logger.info(f"Edge server {args.edge_server_id} is running...")
        
        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Edge server shutting down...")
    finally:
        await edge_server.stop_server()

async def run_client(config, args):
    """Run a single wearable client"""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting wearable client: {args.client_id}")
    
    # Create wearable device with proper config
    device_config = {
        "sensors": {
            "sampling_rate": config.DEVICES.SENSOR_SAMPLING_RATE,
            "accuracy": 0.95
        },
        "hardware": {
            "battery_capacity_mah": config.DEVICES.BATTERY_CAPACITY,
            "processing_power": 1.0
        },
        "training": {
            "local_epochs": config.FL.LOCAL_EPOCHS,
            "batch_size": 32
        },
        "model": {
            "architecture": config.FL.MODEL_TYPE,
            "input_features": config.FL.INPUT_FEATURES,
            "num_classes": config.FL.NUM_CLASSES,
            "lstm_hidden": config.FL.HIDDEN_UNITS,
            "dropout_rate": config.FL.DROPOUT_RATE
        }
    }
    
    device = WearableDevice(
        device_id=args.client_id,
        config=device_config
    )
    
    try:
        # Start data collection and FL participation
        logger.info(f"Wearable device {args.client_id} is running...")
        await device.start_monitoring()
        
    except KeyboardInterrupt:
        logger.info("Wearable device shutting down...")
    # Note: WearableDevice doesn't have a shutdown method

async def run_coordinator(config, args):
    """Run only the FL coordinator"""
    logger = logging.getLogger(__name__)
    logger.info("Starting FL Coordinator")
    
    network_sim = NetworkSimulator(config.NETWORK)
    metrics_collector = MetricsCollector(args.output_dir)
    
    coordinator = FLCoordinator(config, network_sim, metrics_collector)
    
    num_rounds = args.num_rounds or config.FL.NUM_ROUNDS
    
    try:
        # Run coordination logic
        await coordinator.coordinate_rounds(num_rounds)
        logger.info("FL coordination completed")
        
    except KeyboardInterrupt:
        logger.info("FL coordination interrupted")

async def main():
    """Main entry point"""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level, log_file=f"{args.output_dir}/system.log")
    logger = logging.getLogger(__name__)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    try:
        config = Config()
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        sys.exit(1)
    
    # Run the appropriate mode
    try:
        if args.mode == "full":
            await run_full_system(config, args)
        elif args.mode == "edge-server":
            await run_edge_server(config, args)
        elif args.mode == "client":
            await run_client(config, args)
        elif args.mode == "coordinator":
            await run_coordinator(config, args)
        else:
            logger.error(f"Unknown mode: {args.mode}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"System error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
