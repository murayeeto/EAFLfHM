"""
Federated Learning Aggregation Algorithms
Implements various aggregation strategies for model updates
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
import numpy as np
from collections import defaultdict

from ..utils.logger import setup_logger


class BaseAggregator(ABC):
    """Base class for federated learning aggregators"""
    
    def __init__(self):
        self.logger = setup_logger(f"Aggregator-{self.__class__.__name__}")
        self.aggregation_history = []
    
    @abstractmethod
    def aggregate(
        self, 
        model_states: List[Dict[str, torch.Tensor]], 
        weights: List[float],
        client_ids: List[str],
        round_number: int
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client model updates
        
        Args:
            model_states: List of client model state dictionaries
            weights: List of aggregation weights (e.g., data sizes)
            client_ids: List of client identifiers
            round_number: Current federated learning round
            
        Returns:
            Aggregated global model state
        """
        pass


class FedAvgAggregator(BaseAggregator):
    """
    Federated Averaging (FedAvg) aggregation algorithm
    Performs weighted averaging of client model parameters
    """
    
    def __init__(self):
        super().__init__()
        self.name = "FedAvg"
    
    def aggregate(
        self, 
        model_states: List[Dict[str, torch.Tensor]], 
        weights: List[float],
        client_ids: List[str],
        round_number: int
    ) -> Dict[str, torch.Tensor]:
        """Perform FedAvg aggregation"""
        
        if not model_states:
            raise ValueError("No model states provided for aggregation")
        
        if len(model_states) != len(weights) or len(model_states) != len(client_ids):
            raise ValueError("Mismatch in lengths of model_states, weights, and client_ids")
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Initialize aggregated state with zeros
        aggregated_state = {}
        first_state = model_states[0]
        
        for param_name, param_tensor in first_state.items():
            aggregated_state[param_name] = torch.zeros_like(param_tensor)
        
        # Weighted averaging
        for i, (state, weight) in enumerate(zip(model_states, normalized_weights)):
            for param_name, param_tensor in state.items():
                if param_name in aggregated_state:
                    aggregated_state[param_name] += weight * param_tensor
                else:
                    self.logger.warning(
                        f"Parameter {param_name} from client {client_ids[i]} "
                        f"not found in aggregated state"
                    )
        
        # Record aggregation metrics
        self._record_aggregation_metrics(
            model_states, weights, client_ids, round_number, aggregated_state
        )
        
        self.logger.info(
            f"FedAvg aggregation completed for round {round_number} "
            f"with {len(model_states)} clients"
        )
        
        return aggregated_state
    
    def _record_aggregation_metrics(
        self, 
        model_states: List[Dict[str, torch.Tensor]], 
        weights: List[float],
        client_ids: List[str],
        round_number: int,
        aggregated_state: Dict[str, torch.Tensor]
    ):
        """Record aggregation quality metrics"""
        
        try:
            # Calculate parameter diversity (variance across clients)
            param_variances = {}
            
            for param_name in aggregated_state.keys():
                param_values = []
                for state in model_states:
                    if param_name in state:
                        param_values.append(state[param_name].flatten())
                
                if param_values:
                    stacked_params = torch.stack(param_values)
                    variance = torch.var(stacked_params, dim=0).mean().item()
                    param_variances[param_name] = variance
            
            # Calculate weight distribution entropy
            weight_entropy = self._calculate_entropy(weights)
            
            # Store metrics
            aggregation_metrics = {
                "round_number": round_number,
                "num_clients": len(model_states),
                "weight_entropy": weight_entropy,
                "param_variances": param_variances,
                "total_weight": sum(weights),
                "client_ids": client_ids
            }
            
            self.aggregation_history.append(aggregation_metrics)
            
            # Keep only recent history
            if len(self.aggregation_history) > 50:
                self.aggregation_history = self.aggregation_history[-40:]
        
        except Exception as e:
            self.logger.warning(f"Error recording aggregation metrics: {e}")
    
    def _calculate_entropy(self, weights: List[float]) -> float:
        """Calculate entropy of weight distribution"""
        total = sum(weights)
        if total == 0:
            return 0.0
        
        probs = [w / total for w in weights]
        entropy = -sum(p * np.log(p + 1e-8) for p in probs if p > 0)
        return entropy


class AdaptiveAggregator(BaseAggregator):
    """
    Adaptive aggregation algorithm that adjusts weights based on client performance
    """
    
    def __init__(self, adaptation_rate: float = 0.1):
        super().__init__()
        self.name = "Adaptive"
        self.adaptation_rate = adaptation_rate
        self.client_performance_history = defaultdict(list)
        self.client_reliability_scores = defaultdict(lambda: 1.0)
    
    def aggregate(
        self, 
        model_states: List[Dict[str, torch.Tensor]], 
        weights: List[float],
        client_ids: List[str],
        round_number: int
    ) -> Dict[str, torch.Tensor]:
        """Perform adaptive aggregation with performance-based weighting"""
        
        if not model_states:
            raise ValueError("No model states provided for aggregation")
        
        # Calculate adaptive weights
        adaptive_weights = self._calculate_adaptive_weights(weights, client_ids, round_number)
        
        # Normalize adaptive weights
        total_weight = sum(adaptive_weights)
        if total_weight == 0:
            # Fallback to uniform weighting
            adaptive_weights = [1.0] * len(model_states)
            total_weight = len(model_states)
        
        normalized_weights = [w / total_weight for w in adaptive_weights]
        
        # Initialize aggregated state
        aggregated_state = {}
        first_state = model_states[0]
        
        for param_name, param_tensor in first_state.items():
            aggregated_state[param_name] = torch.zeros_like(param_tensor)
        
        # Weighted averaging with adaptive weights
        for i, (state, weight) in enumerate(zip(model_states, normalized_weights)):
            for param_name, param_tensor in state.items():
                if param_name in aggregated_state:
                    aggregated_state[param_name] += weight * param_tensor
        
        # Update client performance tracking
        self._update_client_performance(client_ids, weights, round_number)
        
        self.logger.info(
            f"Adaptive aggregation completed for round {round_number} "
            f"with {len(model_states)} clients (adaptive weights applied)"
        )
        
        return aggregated_state
    
    def _calculate_adaptive_weights(
        self, 
        base_weights: List[float], 
        client_ids: List[str],
        round_number: int
    ) -> List[float]:
        """Calculate adaptive weights based on client reliability and performance"""
        
        adaptive_weights = []
        
        for i, (base_weight, client_id) in enumerate(zip(base_weights, client_ids)):
            # Get client reliability score
            reliability = self.client_reliability_scores[client_id]
            
            # Calculate recency factor (more recent participation is better)
            last_participation = self._get_last_participation_round(client_id)
            recency_factor = 1.0
            if last_participation is not None:
                rounds_since = round_number - last_participation
                recency_factor = max(0.5, 1.0 - (rounds_since * 0.1))
            
            # Calculate performance factor
            performance_factor = self._get_performance_factor(client_id)
            
            # Combine factors
            adaptive_weight = base_weight * reliability * recency_factor * performance_factor
            adaptive_weights.append(adaptive_weight)
        
        return adaptive_weights
    
    def _update_client_performance(
        self, 
        client_ids: List[str], 
        weights: List[float],
        round_number: int
    ):
        """Update client performance history and reliability scores"""
        
        for client_id, weight in zip(client_ids, weights):
            # Record participation
            self.client_performance_history[client_id].append({
                "round": round_number,
                "weight": weight,
                "participation_time": round_number
            })
            
            # Keep limited history
            if len(self.client_performance_history[client_id]) > 20:
                self.client_performance_history[client_id] = (
                    self.client_performance_history[client_id][-15:]
                )
            
            # Update reliability score based on consistency
            self._update_reliability_score(client_id)
    
    def _update_reliability_score(self, client_id: str):
        """Update reliability score for a client"""
        
        history = self.client_performance_history[client_id]
        if len(history) < 3:
            return  # Need more history
        
        # Calculate participation consistency
        recent_rounds = [entry["round"] for entry in history[-10:]]
        if len(recent_rounds) > 1:
            round_gaps = [recent_rounds[i+1] - recent_rounds[i] for i in range(len(recent_rounds)-1)]
            avg_gap = np.mean(round_gaps)
            gap_variance = np.var(round_gaps)
            
            # Lower variance in participation gaps = higher reliability
            consistency_score = 1.0 / (1.0 + gap_variance)
            
            # Update reliability score with exponential moving average
            current_reliability = self.client_reliability_scores[client_id]
            self.client_reliability_scores[client_id] = (
                (1 - self.adaptation_rate) * current_reliability + 
                self.adaptation_rate * consistency_score
            )
    
    def _get_last_participation_round(self, client_id: str) -> Optional[int]:
        """Get the last round when client participated"""
        history = self.client_performance_history[client_id]
        if history:
            return history[-1]["round"]
        return None
    
    def _get_performance_factor(self, client_id: str) -> float:
        """Calculate performance factor based on client history"""
        history = self.client_performance_history[client_id]
        
        if len(history) < 2:
            return 1.0
        
        # Calculate average weight (data contribution)
        recent_weights = [entry["weight"] for entry in history[-5:]]
        avg_weight = np.mean(recent_weights)
        
        # Normalize to [0.5, 1.5] range
        normalized_factor = 0.5 + (avg_weight / max(recent_weights + [1.0]))
        
        return min(1.5, max(0.5, normalized_factor))


class SecureAggregator(BaseAggregator):
    """
    Secure aggregation with privacy preservation
    Implements differential privacy and secure multi-party computation concepts
    """
    
    def __init__(self, noise_multiplier: float = 1.1, max_grad_norm: float = 1.0):
        super().__init__()
        self.name = "SecureAggregation"
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
    
    def aggregate(
        self, 
        model_states: List[Dict[str, torch.Tensor]], 
        weights: List[float],
        client_ids: List[str],
        round_number: int
    ) -> Dict[str, torch.Tensor]:
        """Perform secure aggregation with differential privacy"""
        
        if not model_states:
            raise ValueError("No model states provided for aggregation")
        
        # First perform standard FedAvg
        fedavg_aggregator = FedAvgAggregator()
        aggregated_state = fedavg_aggregator.aggregate(
            model_states, weights, client_ids, round_number
        )
        
        # Apply differential privacy noise
        noisy_state = self._add_differential_privacy_noise(aggregated_state)
        
        # Apply gradient clipping
        clipped_state = self._clip_gradients(noisy_state)
        
        self.logger.info(
            f"Secure aggregation completed for round {round_number} "
            f"with DP noise (σ={self.noise_multiplier})"
        )
        
        return clipped_state
    
    def _add_differential_privacy_noise(
        self, 
        model_state: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Add differential privacy noise to model parameters"""
        
        noisy_state = {}
        
        for param_name, param_tensor in model_state.items():
            # Calculate noise scale based on parameter sensitivity
            noise_scale = self.noise_multiplier * self.max_grad_norm
            
            # Generate Gaussian noise
            noise = torch.normal(
                mean=0.0,
                std=noise_scale,
                size=param_tensor.shape,
                dtype=param_tensor.dtype,
                device=param_tensor.device
            )
            
            # Add noise to parameters
            noisy_state[param_name] = param_tensor + noise
        
        return noisy_state
    
    def _clip_gradients(
        self, 
        model_state: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Apply gradient clipping for privacy protection"""
        
        clipped_state = {}
        
        # Calculate global gradient norm
        total_norm = 0.0
        for param_tensor in model_state.values():
            param_norm = param_tensor.norm(dtype=torch.float32)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        # Apply clipping if necessary
        if total_norm > self.max_grad_norm:
            clip_coef = self.max_grad_norm / (total_norm + 1e-6)
            
            for param_name, param_tensor in model_state.items():
                clipped_state[param_name] = param_tensor * clip_coef
        else:
            clipped_state = model_state
        
        return clipped_state


class RobustAggregator(BaseAggregator):
    """
    Robust aggregation algorithm that filters out malicious or low-quality updates
    """
    
    def __init__(self, outlier_threshold: float = 2.0):
        super().__init__()
        self.name = "RobustAggregation"
        self.outlier_threshold = outlier_threshold
    
    def aggregate(
        self, 
        model_states: List[Dict[str, torch.Tensor]], 
        weights: List[float],
        client_ids: List[str],
        round_number: int
    ) -> Dict[str, torch.Tensor]:
        """Perform robust aggregation with outlier detection"""
        
        if not model_states:
            raise ValueError("No model states provided for aggregation")
        
        # Detect and filter outlier updates
        filtered_states, filtered_weights, filtered_ids = self._filter_outliers(
            model_states, weights, client_ids
        )
        
        if not filtered_states:
            self.logger.warning("All updates filtered as outliers, using original updates")
            filtered_states, filtered_weights, filtered_ids = model_states, weights, client_ids
        
        # Perform standard aggregation on filtered updates
        fedavg_aggregator = FedAvgAggregator()
        aggregated_state = fedavg_aggregator.aggregate(
            filtered_states, filtered_weights, filtered_ids, round_number
        )
        
        filtered_count = len(model_states) - len(filtered_states)
        self.logger.info(
            f"Robust aggregation completed for round {round_number} "
            f"({filtered_count} outliers filtered)"
        )
        
        return aggregated_state
    
    def _filter_outliers(
        self, 
        model_states: List[Dict[str, torch.Tensor]], 
        weights: List[float],
        client_ids: List[str]
    ) -> tuple:
        """Filter outlier model updates based on parameter statistics"""
        
        if len(model_states) < 3:
            return model_states, weights, client_ids  # Need minimum clients for outlier detection
        
        # Calculate parameter norms for each update
        update_norms = []
        for state in model_states:
            total_norm = 0.0
            for param_tensor in state.values():
                total_norm += param_tensor.norm().item() ** 2
            update_norms.append(total_norm ** 0.5)
        
        # Calculate statistics
        norm_array = np.array(update_norms)
        median_norm = np.median(norm_array)
        mad = np.median(np.abs(norm_array - median_norm))  # Median Absolute Deviation
        
        # Filter outliers using MAD-based threshold
        threshold = self.outlier_threshold
        is_outlier = np.abs(norm_array - median_norm) > (threshold * mad)
        
        # Keep non-outlier updates
        filtered_states = [state for i, state in enumerate(model_states) if not is_outlier[i]]
        filtered_weights = [weight for i, weight in enumerate(weights) if not is_outlier[i]]
        filtered_ids = [client_id for i, client_id in enumerate(client_ids) if not is_outlier[i]]
        
        # Log outlier information
        outlier_ids = [client_ids[i] for i in range(len(client_ids)) if is_outlier[i]]
        if outlier_ids:
            self.logger.warning(f"Filtered outlier updates from clients: {outlier_ids}")
        
        return filtered_states, filtered_weights, filtered_ids


def create_aggregator(algorithm_name: str, **kwargs) -> BaseAggregator:
    """
    Factory function to create aggregation algorithms
    
    Args:
        algorithm_name: Name of the aggregation algorithm
        **kwargs: Algorithm-specific parameters
        
    Returns:
        Aggregator instance
    """
    algorithms = {
        "fedavg": FedAvgAggregator,
        "adaptive": AdaptiveAggregator,
        "secure": SecureAggregator,
        "robust": RobustAggregator
    }
    
    if algorithm_name.lower() not in algorithms:
        raise ValueError(f"Unknown aggregation algorithm: {algorithm_name}")
    
    aggregator_class = algorithms[algorithm_name.lower()]
    return aggregator_class(**kwargs)
