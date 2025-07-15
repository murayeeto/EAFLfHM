#!/usr/bin/env python3
"""
Basic tests for the Edge-Aware Federated Learning System
"""

import unittest
import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.config import Config
from models.health_models import HealthCNN, HealthLSTM, HybridHealthModel
from src.wearable.sensor_simulator import SensorSimulator
from src.wearable.battery_manager import BatteryManager
from src.wearable.data_processor import DataProcessor
from src.network.network_simulator import NetworkSimulator
from src.utils.crypto import CryptoManager

class TestConfiguration(unittest.TestCase):
    """Test configuration loading"""
    
    def test_config_loading(self):
        """Test that configuration loads without errors"""
        config = Config()
        self.assertIsNotNone(config.FL)
        self.assertIsNotNone(config.EDGE)
        self.assertIsNotNone(config.NETWORK)
        self.assertIsNotNone(config.DEVICES)

class TestHealthModels(unittest.TestCase):
    """Test health monitoring models"""
    
    def test_cnn_model_creation(self):
        """Test CNN model creation"""
        model = HealthCNN(input_size=100, num_classes=2)
        self.assertIsNotNone(model)
        
        # Test forward pass
        import torch
        x = torch.randn(1, 1, 100)
        output = model(x)
        self.assertEqual(output.shape[1], 2)
    
    def test_lstm_model_creation(self):
        """Test LSTM model creation"""
        model = HealthLSTM(input_size=4, hidden_size=64, num_classes=2)
        self.assertIsNotNone(model)
        
        # Test forward pass
        import torch
        x = torch.randn(1, 50, 4)  # batch_size, seq_len, input_size
        output = model(x)
        self.assertEqual(output.shape[1], 2)
    
    def test_hybrid_model_creation(self):
        """Test hybrid model creation"""
        model = HybridHealthModel(input_size=100, cnn_channels=32, lstm_hidden=64, num_classes=2)
        self.assertIsNotNone(model)
        
        # Test forward pass
        import torch
        x = torch.randn(1, 1, 100)
        output = model(x)
        self.assertEqual(output.shape[1], 2)

class TestSensorSimulator(unittest.TestCase):
    """Test sensor simulation"""
    
    def test_sensor_creation(self):
        """Test sensor simulator creation"""
        sensor = SensorSimulator(device_id="test_device")
        self.assertEqual(sensor.device_id, "test_device")
    
    def test_data_generation(self):
        """Test physiological data generation"""
        sensor = SensorSimulator(device_id="test_device")
        data = sensor.generate_physiological_data()
        
        self.assertIn('heart_rate', data)
        self.assertIn('spo2', data)
        self.assertIn('temperature', data)
        self.assertIn('timestamp', data)
        
        # Check data ranges
        self.assertGreaterEqual(data['heart_rate'], 40)
        self.assertLessEqual(data['heart_rate'], 200)
        self.assertGreaterEqual(data['spo2'], 85)
        self.assertLessEqual(data['spo2'], 100)

class TestBatteryManager(unittest.TestCase):
    """Test battery management"""
    
    def test_battery_creation(self):
        """Test battery manager creation"""
        battery = BatteryManager(initial_capacity=100.0)
        self.assertEqual(battery.capacity, 100.0)
    
    def test_energy_consumption(self):
        """Test energy consumption calculation"""
        battery = BatteryManager(initial_capacity=100.0)
        initial_capacity = battery.capacity
        
        # Consume some energy
        battery.consume_energy(10.0)
        self.assertLess(battery.capacity, initial_capacity)

class TestDataProcessor(unittest.TestCase):
    """Test data processing"""
    
    def test_data_processor_creation(self):
        """Test data processor creation"""
        processor = DataProcessor(window_size=100, overlap=0.5)
        self.assertEqual(processor.window_size, 100)
        self.assertEqual(processor.overlap, 0.5)
    
    def test_feature_extraction(self):
        """Test feature extraction"""
        processor = DataProcessor(window_size=10, overlap=0.0)
        
        # Create sample data
        import numpy as np
        data = {
            'heart_rate': np.random.randint(60, 100, 20),
            'spo2': np.random.randint(95, 100, 20),
            'temperature': np.random.uniform(36.0, 37.5, 20),
            'timestamp': list(range(20))
        }
        
        features = processor.extract_features(data)
        self.assertIsNotNone(features)
        self.assertGreater(len(features), 0)

class TestNetworkSimulator(unittest.TestCase):
    """Test network simulation"""
    
    def test_network_creation(self):
        """Test network simulator creation"""
        config = Config()
        network = NetworkSimulator(config.NETWORK)
        self.assertIsNotNone(network)
    
    def test_latency_calculation(self):
        """Test latency calculation"""
        config = Config()
        network = NetworkSimulator(config.NETWORK)
        
        latency = network.calculate_latency("device_1", "edge_1")
        self.assertGreaterEqual(latency, 0)

class TestCryptoManager(unittest.TestCase):
    """Test cryptographic operations"""
    
    def test_crypto_creation(self):
        """Test crypto manager creation"""
        crypto = CryptoManager()
        self.assertIsNotNone(crypto)
    
    def test_encryption_decryption(self):
        """Test encryption and decryption"""
        crypto = CryptoManager()
        
        # Test data
        original_data = b"test health data"
        
        # Encrypt
        encrypted_data = crypto.encrypt_data(original_data)
        self.assertNotEqual(encrypted_data, original_data)
        
        # Decrypt
        decrypted_data = crypto.decrypt_data(encrypted_data)
        self.assertEqual(decrypted_data, original_data)

class TestAsyncComponents(unittest.TestCase):
    """Test asynchronous components"""
    
    def test_async_operations(self):
        """Test basic async functionality"""
        async def sample_async_function():
            await asyncio.sleep(0.1)
            return "success"
        
        result = asyncio.run(sample_async_function())
        self.assertEqual(result, "success")

def run_tests():
    """Run all tests"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestConfiguration,
        TestHealthModels,
        TestSensorSimulator,
        TestBatteryManager,
        TestDataProcessor,
        TestNetworkSimulator,
        TestCryptoManager,
        TestAsyncComponents
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Return success status
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
