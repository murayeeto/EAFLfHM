"""
Logging utilities for the Edge-Aware Federated Learning system
"""

import logging
import os
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str, 
    log_file: Optional[str] = None, 
    level: int = logging.INFO
) -> logging.Logger:
    """
    Set up a logger with consistent formatting
    
    Args:
        name: Logger name
        log_file: Optional log file path
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def setup_logging(level: str = "INFO", log_file: str = None):
    """
    Setup global logging configuration for the entire system
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    # Convert string level to logging constant
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR
    }
    
    log_level = level_map.get(level.upper(), logging.INFO)
    
    # Create logs directory if it doesn't exist
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    else:
        os.makedirs("logs", exist_ok=True)
        log_file = f"logs/edge_fl_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    # Set up specific loggers for different components
    loggers = [
        'FLCoordinator',
        'EdgeServer', 
        'WearableDevice',
        'NetworkSimulator',
        'MetricsCollector'
    ]
    
    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)
    
    logging.info(f"Logging system initialized - Level: {level}, File: {log_file}")


class PerformanceLogger:
    """Logger for performance metrics"""
    
    def __init__(self, log_file: str = "logs/performance.log"):
        self.logger = setup_logger("Performance", log_file)
        
    def log_training_metrics(
        self, 
        device_id: str, 
        round_num: int, 
        training_time: float,
        loss: float,
        accuracy: float
    ):
        """Log training performance metrics"""
        self.logger.info(
            f"TRAINING - Device: {device_id}, Round: {round_num}, "
            f"Time: {training_time:.2f}s, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}"
        )
    
    def log_communication_metrics(
        self, 
        device_id: str, 
        data_size_mb: float, 
        latency_ms: float
    ):
        """Log communication performance metrics"""
        self.logger.info(
            f"COMMUNICATION - Device: {device_id}, "
            f"Size: {data_size_mb:.2f}MB, Latency: {latency_ms:.2f}ms"
        )
    
    def log_energy_metrics(
        self, 
        device_id: str, 
        power_consumption_w: float, 
        battery_level: float
    ):
        """Log energy consumption metrics"""
        self.logger.info(
            f"ENERGY - Device: {device_id}, "
            f"Power: {power_consumption_w:.3f}W, Battery: {battery_level:.1f}%"
        )
