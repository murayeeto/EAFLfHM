#!/usr/bin/env python3
"""
Simple example demonstrating the basic usage of the Edge-Aware FL system
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.config import Config
from src.wearable.device_simulator import WearableDevice
from src.edge.edge_server import EdgeServer
from src.federated.fl_coordinator import FLCoordinator
from src.network.network_simulator import NetworkSimulator
from src.evaluation.metrics_collector import MetricsCollector
from src.utils.logger import setup_logging

async def simple_fl_demo():
    """
    Simple demonstration of federated learning with 3 devices and 1 edge server
    """
    print("🚀 Starting Simple FL Demo")
    print("=" * 50)
    
    # Setup logging
    setup_logging(level="INFO")
    
    # Load configuration
    config = Config()
    
    # Create output directory
    output_dir = "examples/demo_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize components
    network_sim = NetworkSimulator(config.NETWORK)
    metrics_collector = MetricsCollector(output_dir)
    
    print("📊 Creating FL Coordinator...")
    coordinator = FLCoordinator(config, network_sim, metrics_collector)
    
    # Run a small experiment
    print("🔬 Running FL experiment with 3 devices for 5 rounds...")
    
    try:
        # Create experiment configuration
        from src.federated.fl_coordinator import ExperimentConfig
        experiment_config = ExperimentConfig(
            num_rounds=5,
            num_edge_servers=2,
            num_devices=3,
            experiment_duration_hours=0.5,
            data_distribution='iid',
            device_failure_rate=0.05,
            network_conditions='stable'
        )
        
        results = await coordinator.run_experiment(experiment_config)
        
        print("✅ Demo completed successfully!")
        print("📁 Results saved to:", output_dir)
        
        if results:
            print("\\n📈 Quick Results:")
            print(f"   Experiment ID: {getattr(results, 'experiment_id', 'N/A')}")
            print(f"   Final accuracy: {getattr(results, 'final_accuracy', 0.0):.4f}")
            print(f"   Convergence round: {getattr(results, 'convergence_round', 'N/A')}")
            print(f"   Average round time: {getattr(results, 'average_round_time', 0.0):.2f}s")
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        raise

async def single_device_demo():
    """
    Demonstrate a single wearable device
    """
    print("\\n👤 Single Device Demo")
    print("=" * 30)
    
    config = Config()
    
    # Create proper device config structure with hardcoded demo values
    device_config = {
        "sensors": {
            "sampling_rate": 1.0,
            "accuracy": 0.95
        },
        "hardware": {
            "battery_capacity_mah": 300,
            "processing_power": 1.0
        },
        "training": {
            "local_epochs": 5,
            "batch_size": 32
        },
        "model": {
            "architecture": "hybrid",
            "input_features": 3,
            "num_classes": 2,
            "lstm_hidden": 64,
            "dropout_rate": 0.3
        }
    }
    
    # Create a single device
    device = WearableDevice(device_id="demo_device", config=device_config)
    
    print("🔄 Device created successfully!")
    
    print("📊 Generating sample data...")
    # Simulate data collection for a few iterations
    for i in range(5):
        data = device.sensor_simulator.generate_physiological_data()
        print(f"   Sample {i+1}: HR={data['heart_rate']}, SpO2={data['spo2']:.1f}%")
        await asyncio.sleep(1)
    
    print("🔋 Battery status:", f"{device.battery_manager.get_battery_level():.1f}%")
    
    print("✅ Device demo completed")

async def edge_server_demo():
    """
    Demonstrate edge server functionality
    """
    print("\\n🏢 Edge Server Demo")
    print("=" * 30)
    
    config = Config()
    
    # Create proper edge server config structure with hardcoded demo values
    edge_config = {
        "max_clients": 50,
        "min_clients": 3,
        "max_wait_time": 120,
        "aggregation_algorithm": "fedavg",
        "model": {
            "architecture": "hybrid",
            "input_features": 3,
            "num_classes": 2,
            "lstm_hidden": 64,
            "dropout_rate": 0.3
        }
    }
    
    # Create edge server
    edge_server = EdgeServer(
        server_id="demo_edge",
        config=edge_config
    )
    
    print("🚀 Starting edge server...")
    await edge_server.start_server()
    
    print("📊 Server status:")
    print(f"   Server ID: {edge_server.server_id}")
    print(f"   Max clients: {edge_server.max_clients}")
    print(f"   Active clients: {len(edge_server.client_manager.clients)}")
    
    # Simulate brief operation
    await asyncio.sleep(2)
    
    print("🛑 Stopping edge server...")
    await edge_server.stop_server()
    print("✅ Edge server demo completed")

def print_system_info():
    """Print system information and capabilities"""
    print("🔍 System Information")
    print("=" * 30)
    
    config = Config()
    
    print(f"FL Configuration:")
    print(f"   Model: {config.FL.MODEL_TYPE}")
    print(f"   Rounds: {config.FL.NUM_ROUNDS}")
    print(f"   Clients per round: {config.FL.CLIENTS_PER_ROUND}")
    print(f"   Learning rate: {config.FL.LEARNING_RATE}")
    
    print(f"\\nDevice Configuration:")
    print(f"   Battery capacity: {config.DEVICES.BATTERY_CAPACITY} mAh")
    print(f"   Sampling rate: {config.DEVICES.SENSOR_SAMPLING_RATE} Hz")
    print(f"   Data window: {config.DEVICES.DATA_WINDOW_SIZE} samples")
    
    print(f"\\nNetwork Configuration:")
    print(f"   Base latency: {config.NETWORK.BASE_LATENCY} ms")
    print(f"   Bandwidth: {config.NETWORK.BANDWIDTH} Mbps")
    print(f"   Edge servers: {config.NETWORK.NUM_EDGE_SERVERS}")

async def main():
    """Main demo function"""
    print("🎯 Edge-Aware Federated Learning - Interactive Demo")
    print("=" * 60)
    
    # Print system info
    print_system_info()
    
    try:
        # Run individual component demos
        await single_device_demo()
        await edge_server_demo()
        
        # Run the main FL demo
        await simple_fl_demo()
        
        print("\\n🎉 All demos completed successfully!")
        print("\\n🔗 Next steps:")
        print("   1. Run full experiments: python scripts/run_experiments.py")
        print("   2. Visualize results: python scripts/visualize_results.py")
        print("   3. Customize config: edit config/config.py")
        
    except KeyboardInterrupt:
        print("\\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\\n❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
