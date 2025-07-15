"""
Metrics Collector for Federated Learning Evaluation
Collects and analyzes performance metrics from the FL system
"""

import time
import json
import csv
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from ..utils.logger import setup_logger


@dataclass
class TrainingMetrics:
    """Training performance metrics"""
    device_id: str
    round_number: int
    training_time: float
    loss: float
    accuracy: float
    data_size: int
    energy_consumption: float
    timestamp: datetime


@dataclass
class CommunicationMetrics:
    """Communication performance metrics"""
    device_id: str
    edge_server_id: str
    round_number: int
    upload_size_mb: float
    download_size_mb: float
    upload_time: float
    download_time: float
    latency_ms: float
    packet_loss_rate: float
    timestamp: datetime


@dataclass
class SystemMetrics:
    """System-wide performance metrics"""
    timestamp: datetime
    total_devices: int
    active_devices: int
    total_edge_servers: int
    active_edge_servers: int
    global_accuracy: float
    round_number: int
    convergence_status: str


class MetricsCollector:
    """
    Comprehensive metrics collection and analysis for FL experiments
    """
    
    def __init__(self, output_dir: str = "results"):
        self.logger = setup_logger("MetricsCollector")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Metrics storage
        self.training_metrics: List[TrainingMetrics] = []
        self.communication_metrics: List[CommunicationMetrics] = []
        self.system_metrics: List[SystemMetrics] = []
        
        # Experiment tracking
        self.current_experiment_id = None
        self.experiment_start_time = None
        self.experiment_config = None
        
        # Real-time aggregations
        self.round_summaries = {}
        self.device_summaries = defaultdict(dict)
        self.server_summaries = defaultdict(dict)
        
        self.logger.info(f"Metrics collector initialized - Output: {self.output_dir}")
    
    def start_experiment(self, experiment_id: str, config: Any):
        """Start metrics collection for a new experiment"""
        self.current_experiment_id = experiment_id
        self.experiment_start_time = datetime.now()
        self.experiment_config = config
        
        # Clear previous metrics
        self.training_metrics.clear()
        self.communication_metrics.clear()
        self.system_metrics.clear()
        self.round_summaries.clear()
        self.device_summaries.clear()
        self.server_summaries.clear()
        
        self.logger.info(f"Started metrics collection for experiment {experiment_id}")
    
    def record_training_metrics(
        self,
        device_id: str,
        round_number: int,
        training_time: float,
        loss: float,
        accuracy: float,
        data_size: int,
        energy_consumption: float
    ):
        """Record training metrics from a device"""
        metrics = TrainingMetrics(
            device_id=device_id,
            round_number=round_number,
            training_time=training_time,
            loss=loss,
            accuracy=accuracy,
            data_size=data_size,
            energy_consumption=energy_consumption,
            timestamp=datetime.now()
        )
        
        self.training_metrics.append(metrics)
        self._update_device_summary(device_id, metrics)
        self._update_round_summary(round_number, "training", metrics)
    
    def record_communication_metrics(
        self,
        device_id: str,
        edge_server_id: str,
        round_number: int,
        upload_size_mb: float,
        download_size_mb: float,
        upload_time: float,
        download_time: float,
        latency_ms: float,
        packet_loss_rate: float
    ):
        """Record communication metrics"""
        metrics = CommunicationMetrics(
            device_id=device_id,
            edge_server_id=edge_server_id,
            round_number=round_number,
            upload_size_mb=upload_size_mb,
            download_size_mb=download_size_mb,
            upload_time=upload_time,
            download_time=download_time,
            latency_ms=latency_ms,
            packet_loss_rate=packet_loss_rate,
            timestamp=datetime.now()
        )
        
        self.communication_metrics.append(metrics)
        self._update_server_summary(edge_server_id, metrics)
        self._update_round_summary(round_number, "communication", metrics)
    
    def record_system_metrics(
        self,
        total_devices: int,
        active_devices: int,
        total_edge_servers: int,
        active_edge_servers: int,
        global_accuracy: float,
        round_number: int,
        convergence_status: str = "training"
    ):
        """Record system-wide metrics"""
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            total_devices=total_devices,
            active_devices=active_devices,
            total_edge_servers=total_edge_servers,
            active_edge_servers=active_edge_servers,
            global_accuracy=global_accuracy,
            round_number=round_number,
            convergence_status=convergence_status
        )
        
        self.system_metrics.append(metrics)
        self._update_round_summary(round_number, "system", metrics)
    
    def _update_device_summary(self, device_id: str, metrics: TrainingMetrics):
        """Update device-level summary statistics"""
        if device_id not in self.device_summaries:
            self.device_summaries[device_id] = {
                "total_rounds": 0,
                "total_training_time": 0.0,
                "total_energy": 0.0,
                "total_data_points": 0,
                "avg_accuracy": 0.0,
                "avg_loss": 0.0,
                "accuracy_history": [],
                "loss_history": []
            }
        
        summary = self.device_summaries[device_id]
        summary["total_rounds"] += 1
        summary["total_training_time"] += metrics.training_time
        summary["total_energy"] += metrics.energy_consumption
        summary["total_data_points"] += metrics.data_size
        summary["accuracy_history"].append(metrics.accuracy)
        summary["loss_history"].append(metrics.loss)
        
        # Update averages
        summary["avg_accuracy"] = np.mean(summary["accuracy_history"])
        summary["avg_loss"] = np.mean(summary["loss_history"])
    
    def _update_server_summary(self, server_id: str, metrics: CommunicationMetrics):
        """Update edge server summary statistics"""
        if server_id not in self.server_summaries:
            self.server_summaries[server_id] = {
                "total_communications": 0,
                "total_upload_mb": 0.0,
                "total_download_mb": 0.0,
                "avg_latency": 0.0,
                "avg_packet_loss": 0.0,
                "latency_history": [],
                "packet_loss_history": []
            }
        
        summary = self.server_summaries[server_id]
        summary["total_communications"] += 1
        summary["total_upload_mb"] += metrics.upload_size_mb
        summary["total_download_mb"] += metrics.download_size_mb
        summary["latency_history"].append(metrics.latency_ms)
        summary["packet_loss_history"].append(metrics.packet_loss_rate)
        
        # Update averages
        summary["avg_latency"] = np.mean(summary["latency_history"])
        summary["avg_packet_loss"] = np.mean(summary["packet_loss_history"])
    
    def _update_round_summary(self, round_number: int, metric_type: str, metrics: Any):
        """Update round-level summary statistics"""
        if round_number not in self.round_summaries:
            self.round_summaries[round_number] = {
                "round_number": round_number,
                "participating_devices": set(),
                "participating_servers": set(),
                "total_training_time": 0.0,
                "total_communication_time": 0.0,
                "total_energy": 0.0,
                "avg_accuracy": 0.0,
                "global_accuracy": 0.0,
                "convergence_status": "training"
            }
        
        summary = self.round_summaries[round_number]
        
        if metric_type == "training":
            summary["participating_devices"].add(metrics.device_id)
            summary["total_training_time"] += metrics.training_time
            summary["total_energy"] += metrics.energy_consumption
        elif metric_type == "communication":
            summary["participating_devices"].add(metrics.device_id)
            summary["participating_servers"].add(metrics.edge_server_id)
            summary["total_communication_time"] += (metrics.upload_time + metrics.download_time)
        elif metric_type == "system":
            summary["global_accuracy"] = metrics.global_accuracy
            summary["convergence_status"] = metrics.convergence_status
    
    def analyze_convergence(self) -> Dict[str, Any]:
        """Analyze model convergence characteristics"""
        if not self.system_metrics:
            return {"error": "No system metrics available"}
        
        # Extract accuracy progression
        rounds = [m.round_number for m in self.system_metrics]
        accuracies = [m.global_accuracy for m in self.system_metrics]
        
        if len(accuracies) < 3:
            return {"error": "Insufficient data for convergence analysis"}
        
        # Calculate convergence metrics
        final_accuracy = accuracies[-1]
        max_accuracy = max(accuracies)
        
        # Find convergence point (when accuracy stops improving significantly)
        convergence_round = None
        accuracy_threshold = 0.95 * max_accuracy
        stability_window = 5
        
        for i in range(stability_window, len(accuracies)):
            recent_accuracies = accuracies[i-stability_window:i]
            if (min(recent_accuracies) >= accuracy_threshold and
                max(recent_accuracies) - min(recent_accuracies) < 0.01):
                convergence_round = rounds[i]
                break
        
        # Calculate convergence rate
        if convergence_round:
            convergence_rate = (max_accuracy - accuracies[0]) / convergence_round
        else:
            convergence_rate = (final_accuracy - accuracies[0]) / len(accuracies)
        
        return {
            "final_accuracy": final_accuracy,
            "max_accuracy": max_accuracy,
            "convergence_round": convergence_round,
            "convergence_rate": convergence_rate,
            "accuracy_improvement": final_accuracy - accuracies[0],
            "stability_achieved": convergence_round is not None,
            "rounds_to_convergence": convergence_round or len(rounds)
        }
    
    def analyze_communication_efficiency(self) -> Dict[str, Any]:
        """Analyze communication efficiency metrics"""
        if not self.communication_metrics:
            return {"error": "No communication metrics available"}
        
        # Aggregate communication statistics
        total_upload = sum(m.upload_size_mb for m in self.communication_metrics)
        total_download = sum(m.download_size_mb for m in self.communication_metrics)
        total_communication = total_upload + total_download
        
        avg_latency = np.mean([m.latency_ms for m in self.communication_metrics])
        avg_packet_loss = np.mean([m.packet_loss_rate for m in self.communication_metrics])
        
        # Communication per round
        round_communication = defaultdict(float)
        for metrics in self.communication_metrics:
            round_communication[metrics.round_number] += (
                metrics.upload_size_mb + metrics.download_size_mb
            )
        
        avg_communication_per_round = np.mean(list(round_communication.values()))
        
        # Efficiency per server
        server_efficiency = {}
        for server_id, summary in self.server_summaries.items():
            total_data = summary["total_upload_mb"] + summary["total_download_mb"]
            total_comms = summary["total_communications"]
            efficiency = total_data / max(1, total_comms)  # MB per communication
            server_efficiency[server_id] = efficiency
        
        return {
            "total_communication_mb": total_communication,
            "total_upload_mb": total_upload,
            "total_download_mb": total_download,
            "average_latency_ms": avg_latency,
            "average_packet_loss": avg_packet_loss,
            "communication_per_round_mb": avg_communication_per_round,
            "server_efficiency": server_efficiency,
            "total_communications": len(self.communication_metrics)
        }
    
    def analyze_energy_consumption(self) -> Dict[str, Any]:
        """Analyze energy consumption patterns"""
        if not self.training_metrics:
            return {"error": "No training metrics available"}
        
        # Total energy consumption
        total_energy = sum(m.energy_consumption for m in self.training_metrics)
        
        # Energy per device
        device_energy = defaultdict(float)
        for metrics in self.training_metrics:
            device_energy[metrics.device_id] += metrics.energy_consumption
        
        avg_energy_per_device = np.mean(list(device_energy.values()))
        
        # Energy per round
        round_energy = defaultdict(float)
        for metrics in self.training_metrics:
            round_energy[metrics.round_number] += metrics.energy_consumption
        
        avg_energy_per_round = np.mean(list(round_energy.values()))
        
        # Energy efficiency (accuracy per unit energy)
        if self.system_metrics:
            final_accuracy = self.system_metrics[-1].global_accuracy
            energy_efficiency = final_accuracy / max(1, total_energy)
        else:
            energy_efficiency = 0.0
        
        return {
            "total_energy_consumption": total_energy,
            "average_energy_per_device": avg_energy_per_device,
            "average_energy_per_round": avg_energy_per_round,
            "energy_efficiency": energy_efficiency,
            "device_energy_distribution": dict(device_energy),
            "round_energy_distribution": dict(round_energy)
        }
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        report = {
            "experiment_id": self.current_experiment_id,
            "experiment_duration": (
                (datetime.now() - self.experiment_start_time).total_seconds() 
                if self.experiment_start_time else 0
            ),
            "data_summary": {
                "training_records": len(self.training_metrics),
                "communication_records": len(self.communication_metrics),
                "system_records": len(self.system_metrics)
            }
        }
        
        # Add analysis sections
        report["convergence_analysis"] = self.analyze_convergence()
        report["communication_analysis"] = self.analyze_communication_efficiency()
        report["energy_analysis"] = self.analyze_energy_consumption()
        
        # Add device and server summaries
        report["device_summaries"] = dict(self.device_summaries)
        report["server_summaries"] = dict(self.server_summaries)
        report["round_summaries"] = {
            k: {**v, "participating_devices": len(v["participating_devices"]), 
                "participating_servers": len(v["participating_servers"])}
            for k, v in self.round_summaries.items()
        }
        
        return report
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = self.current_experiment_id or "unknown"
        
        if format.lower() == "json":
            filename = f"{experiment_id}_{timestamp}_metrics.json"
            filepath = self.output_dir / filename
            
            export_data = {
                "experiment_info": {
                    "experiment_id": experiment_id,
                    "start_time": self.experiment_start_time.isoformat() if self.experiment_start_time else None,
                    "export_time": datetime.now().isoformat(),
                    "config": asdict(self.experiment_config) if self.experiment_config else None
                },
                "training_metrics": [asdict(m) for m in self.training_metrics],
                "communication_metrics": [asdict(m) for m in self.communication_metrics],
                "system_metrics": [asdict(m) for m in self.system_metrics],
                "analysis": self.generate_comprehensive_report()
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        
        elif format.lower() == "csv":
            # Export each metric type to separate CSV files
            base_filename = f"{experiment_id}_{timestamp}"
            
            # Training metrics CSV
            if self.training_metrics:
                training_file = self.output_dir / f"{base_filename}_training.csv"
                with open(training_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=asdict(self.training_metrics[0]).keys())
                    writer.writeheader()
                    for metrics in self.training_metrics:
                        writer.writerow(asdict(metrics))
            
            # Communication metrics CSV
            if self.communication_metrics:
                comm_file = self.output_dir / f"{base_filename}_communication.csv"
                with open(comm_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=asdict(self.communication_metrics[0]).keys())
                    writer.writeheader()
                    for metrics in self.communication_metrics:
                        writer.writerow(asdict(metrics))
            
            # System metrics CSV
            if self.system_metrics:
                system_file = self.output_dir / f"{base_filename}_system.csv"
                with open(system_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=asdict(self.system_metrics[0]).keys())
                    writer.writeheader()
                    for metrics in self.system_metrics:
                        writer.writerow(asdict(metrics))
            
            filepath = str(self.output_dir / f"{base_filename}_*.csv")
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        self.logger.info(f"Metrics exported to {filepath}")
        return str(filepath)
    
    def generate_visualizations(self) -> List[str]:
        """Generate visualization plots"""
        if not any([self.training_metrics, self.communication_metrics, self.system_metrics]):
            self.logger.warning("No data available for visualization")
            return []
        
        plt.style.use('seaborn-v0_8')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = self.current_experiment_id or "unknown"
        plot_files = []
        
        # 1. Convergence plot
        if self.system_metrics:
            plt.figure(figsize=(10, 6))
            rounds = [m.round_number for m in self.system_metrics]
            accuracies = [m.global_accuracy for m in self.system_metrics]
            
            plt.plot(rounds, accuracies, 'b-', linewidth=2, marker='o', markersize=4)
            plt.xlabel('Federated Learning Round')
            plt.ylabel('Global Model Accuracy')
            plt.title('Model Convergence Over Time')
            plt.grid(True, alpha=0.3)
            
            convergence_file = self.output_dir / f"{experiment_id}_{timestamp}_convergence.png"
            plt.savefig(convergence_file, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files.append(str(convergence_file))
        
        # 2. Communication overhead plot
        if self.communication_metrics:
            plt.figure(figsize=(12, 8))
            
            # Aggregate by round
            round_comm = defaultdict(float)
            for m in self.communication_metrics:
                round_comm[m.round_number] += (m.upload_size_mb + m.download_size_mb)
            
            rounds = sorted(round_comm.keys())
            comm_sizes = [round_comm[r] for r in rounds]
            
            plt.subplot(2, 2, 1)
            plt.plot(rounds, comm_sizes, 'g-', linewidth=2)
            plt.xlabel('Round')
            plt.ylabel('Communication (MB)')
            plt.title('Communication Overhead per Round')
            
            # Latency distribution
            plt.subplot(2, 2, 2)
            latencies = [m.latency_ms for m in self.communication_metrics]
            plt.hist(latencies, bins=30, alpha=0.7, color='orange')
            plt.xlabel('Latency (ms)')
            plt.ylabel('Frequency')
            plt.title('Latency Distribution')
            
            # Server communication distribution
            plt.subplot(2, 2, 3)
            server_comm = defaultdict(float)
            for m in self.communication_metrics:
                server_comm[m.edge_server_id] += (m.upload_size_mb + m.download_size_mb)
            
            servers = list(server_comm.keys())
            comm_values = list(server_comm.values())
            plt.bar(servers, comm_values, color='purple', alpha=0.7)
            plt.xlabel('Edge Server')
            plt.ylabel('Total Communication (MB)')
            plt.title('Communication per Edge Server')
            plt.xticks(rotation=45)
            
            # Packet loss over time
            plt.subplot(2, 2, 4)
            timestamps = [m.timestamp for m in self.communication_metrics]
            packet_losses = [m.packet_loss_rate * 100 for m in self.communication_metrics]
            plt.scatter(timestamps, packet_losses, alpha=0.6, color='red', s=10)
            plt.xlabel('Time')
            plt.ylabel('Packet Loss (%)')
            plt.title('Packet Loss Over Time')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            comm_file = self.output_dir / f"{experiment_id}_{timestamp}_communication.png"
            plt.savefig(comm_file, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files.append(str(comm_file))
        
        # 3. Energy consumption analysis
        if self.training_metrics:
            plt.figure(figsize=(12, 6))
            
            # Energy per device
            device_energy = defaultdict(float)
            for m in self.training_metrics:
                device_energy[m.device_id] += m.energy_consumption
            
            plt.subplot(1, 2, 1)
            devices = list(device_energy.keys())[:20]  # Show top 20 devices
            energies = [device_energy[d] for d in devices]
            plt.bar(range(len(devices)), energies, color='coral')
            plt.xlabel('Device Index')
            plt.ylabel('Total Energy Consumption')
            plt.title('Energy Consumption per Device (Top 20)')
            
            # Energy over time
            plt.subplot(1, 2, 2)
            round_energy = defaultdict(float)
            for m in self.training_metrics:
                round_energy[m.round_number] += m.energy_consumption
            
            rounds = sorted(round_energy.keys())
            energies = [round_energy[r] for r in rounds]
            plt.plot(rounds, energies, 'r-', linewidth=2, marker='s', markersize=4)
            plt.xlabel('Round')
            plt.ylabel('Total Energy Consumption')
            plt.title('Energy Consumption per Round')
            
            plt.tight_layout()
            energy_file = self.output_dir / f"{experiment_id}_{timestamp}_energy.png"
            plt.savefig(energy_file, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files.append(str(energy_file))
        
        self.logger.info(f"Generated {len(plot_files)} visualization plots")
        return plot_files
