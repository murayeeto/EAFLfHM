"""
Neural Network Models for Health Monitoring
Implements lightweight 1D CNN and LSTM architectures for wearable devices
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class HealthCNN1D(nn.Module):
    """
    Lightweight 1D CNN for physiological signal analysis
    Optimized for wearable device deployment
    """
    
    def __init__(
        self, 
        input_features: int = 3,
        num_classes: int = 2,
        hidden_units: int = 64,
        dropout_rate: float = 0.3,
        sequence_length: int = 60
    ):
        super(HealthCNN1D, self).__init__()
        
        self.input_features = input_features
        self.num_classes = num_classes
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(input_features, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, hidden_units, kernel_size=3, padding=1)
        
        # Pooling layers
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate flattened size after convolutions
        self._calculate_conv_output_size(sequence_length)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.conv_output_size, hidden_units)
        self.fc2 = nn.Linear(hidden_units, 32)
        self.fc3 = nn.Linear(32, num_classes)
        
        # Regularization
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.batch_norm3 = nn.BatchNorm1d(hidden_units)
        
    def _calculate_conv_output_size(self, sequence_length: int):
        """Calculate the output size after convolution and pooling layers"""
        # For very short sequences (like single feature vectors), adjust the calculation
        if sequence_length <= 4:
            # No pooling for very short sequences, just use conv outputs
            self.conv_output_size = 64 * sequence_length  # 64 is the output channels of conv3
            self.use_pooling = False
        else:
            # After conv1 + pool: length // 2
            # After conv2 + pool: length // 4  
            # After conv3 + pool: length // 8
            conv_output_length = sequence_length // 8
            self.conv_output_size = 64 * conv_output_length
            self.use_pooling = True
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_features, sequence_length)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Convolutional layers with conditional pooling
        if hasattr(self, 'use_pooling') and not self.use_pooling:
            # For very short sequences, skip pooling
            x = F.relu(self.batch_norm1(self.conv1(x)))
            x = F.relu(self.batch_norm2(self.conv2(x)))
            x = F.relu(self.batch_norm3(self.conv3(x)))
        else:
            # Normal pooling for longer sequences
            x = self.pool(F.relu(self.batch_norm1(self.conv1(x))))
            x = self.pool(F.relu(self.batch_norm2(self.conv2(x))))
            x = self.pool(F.relu(self.batch_norm3(self.conv3(x))))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class HealthLSTM(nn.Module):
    """
    Lightweight LSTM for sequential physiological data analysis
    """
    
    def __init__(
        self,
        input_features: int = 3,
        num_classes: int = 2,
        hidden_units: int = 64,
        num_layers: int = 2,
        dropout_rate: float = 0.3,
        bidirectional: bool = False
    ):
        super(HealthLSTM, self).__init__()
        
        self.input_features = input_features
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_units,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate LSTM output size
        lstm_output_size = hidden_units * 2 if bidirectional else hidden_units
        
        # Fully connected layers
        self.fc1 = nn.Linear(lstm_output_size, hidden_units)
        self.fc2 = nn.Linear(hidden_units, 32)
        self.fc3 = nn.Linear(32, num_classes)
        
        # Regularization
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_features)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last output for classification
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            last_output = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            last_output = hidden[-1]
        
        # Fully connected layers
        x = F.relu(self.fc1(last_output))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class HybridCNNLSTM(nn.Module):
    """
    Hybrid model combining CNN feature extraction with LSTM temporal modeling
    """
    
    def __init__(
        self,
        input_features: int = 3,
        num_classes: int = 2,
        cnn_channels: int = 32,
        lstm_hidden: int = 64,
        dropout_rate: float = 0.3,
        sequence_length: int = 60
    ):
        super(HybridCNNLSTM, self).__init__()
        
        self.sequence_length = sequence_length
        
        # CNN feature extractor
        self.conv1 = nn.Conv1d(input_features, cnn_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(cnn_channels, cnn_channels * 2, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=cnn_channels * 2,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_features, sequence_length)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # CNN feature extraction with conditional pooling
        if hasattr(self, 'sequence_length') and self.sequence_length <= 2:
            # For very short sequences, skip pooling to avoid zero-size output
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
        else:
            # Normal pooling for longer sequences
            x = F.relu(self.conv1(x))
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            x = self.pool(x)
        
        # Reshape for LSTM (batch_size, seq_len, features)
        x = x.transpose(1, 2)
        
        # LSTM temporal modeling
        lstm_out, (hidden, _) = self.lstm(x)
        
        # Use last hidden state for classification
        x = hidden[-1]
        
        # Classification
        x = self.classifier(x)
        
        return x


def create_model(
    architecture: str = "1d_cnn",
    input_features: int = 3,
    num_classes: int = 2,
    **kwargs
) -> nn.Module:
    """
    Factory function to create neural network models
    
    Args:
        architecture: Model architecture ("1d_cnn", "lstm", "hybrid")
        input_features: Number of input features
        num_classes: Number of output classes
        **kwargs: Additional model parameters
        
    Returns:
        PyTorch model instance
    """
    if architecture == "1d_cnn":
        return HealthCNN1D(input_features, num_classes, **kwargs)
    elif architecture == "lstm":
        return HealthLSTM(input_features, num_classes, **kwargs)
    elif architecture == "hybrid":
        return HybridCNNLSTM(input_features, num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: nn.Module) -> float:
    """Calculate model size in MB"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb
