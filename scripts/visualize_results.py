#!/usr/bin/env python3
"""
Script to visualize and analyze federated learning experiment results
"""

import os
import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_experiment_results(results_dir):
    """Load experiment results from JSON files"""
    results = {}
    
    for experiment_dir in os.listdir(results_dir):
        experiment_path = os.path.join(results_dir, experiment_dir)
        if os.path.isdir(experiment_path):
            # Look for results.json or metrics.json
            for filename in ['results.json', 'metrics.json', 'experiment_results.json']:
                file_path = os.path.join(experiment_path, filename)
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'r') as f:
                            results[experiment_dir] = json.load(f)
                        break
                    except Exception as e:
                        print(f"Warning: Could not load {file_path}: {e}")
    
    return results

def plot_training_curves(results, output_dir):
    """Plot training accuracy and loss curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    for exp_name, data in results.items():
        if 'rounds' in data and 'accuracy' in data:
            rounds = data['rounds']
            accuracy = data['accuracy']
            loss = data.get('loss', [])
            
            ax1.plot(rounds, accuracy, marker='o', label=exp_name, linewidth=2)
            if loss:
                ax2.plot(rounds, loss, marker='s', label=exp_name, linewidth=2)
    
    ax1.set_xlabel('FL Round')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Training Accuracy Over FL Rounds')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('FL Round')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss Over FL Rounds')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_communication_overhead(results, output_dir):
    """Plot communication overhead comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    experiments = []
    comm_overhead = []
    
    for exp_name, data in results.items():
        if 'communication_overhead' in data:
            experiments.append(exp_name)
            # Get average communication overhead
            if isinstance(data['communication_overhead'], list):
                comm_overhead.append(np.mean(data['communication_overhead']))
            else:
                comm_overhead.append(data['communication_overhead'])
    
    if experiments:
        bars = ax.bar(experiments, comm_overhead, color=sns.color_palette("viridis", len(experiments)))
        ax.set_xlabel('Experiment')
        ax.set_ylabel('Communication Overhead (MB)')
        ax.set_title('Average Communication Overhead by Experiment')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, comm_overhead):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(comm_overhead)*0.01,
                   f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'communication_overhead.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_energy_consumption(results, output_dir):
    """Plot energy consumption analysis"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    experiments = []
    total_energy = []
    avg_energy_per_round = []
    
    for exp_name, data in results.items():
        if 'energy_consumption' in data:
            experiments.append(exp_name)
            
            if isinstance(data['energy_consumption'], list):
                total_energy.append(sum(data['energy_consumption']))
                avg_energy_per_round.append(np.mean(data['energy_consumption']))
            else:
                total_energy.append(data['energy_consumption'])
                avg_energy_per_round.append(data['energy_consumption'])
    
    if experiments:
        # Total energy consumption
        ax1.bar(experiments, total_energy, color=sns.color_palette("plasma", len(experiments)))
        ax1.set_xlabel('Experiment')
        ax1.set_ylabel('Total Energy Consumption (J)')
        ax1.set_title('Total Energy Consumption by Experiment')
        ax1.tick_params(axis='x', rotation=45)
        
        # Average energy per round
        ax2.bar(experiments, avg_energy_per_round, color=sns.color_palette("cividis", len(experiments)))
        ax2.set_xlabel('Experiment')
        ax2.set_ylabel('Average Energy per Round (J)')
        ax2.set_title('Average Energy Consumption per Round')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'energy_consumption.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_latency_analysis(results, output_dir):
    """Plot training and communication latency"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    experiments = []
    training_latency = []
    comm_latency = []
    
    for exp_name, data in results.items():
        experiments.append(exp_name)
        
        # Training latency
        if 'training_latency' in data:
            if isinstance(data['training_latency'], list):
                training_latency.append(np.mean(data['training_latency']))
            else:
                training_latency.append(data['training_latency'])
        else:
            training_latency.append(0)
        
        # Communication latency
        if 'communication_latency' in data:
            if isinstance(data['communication_latency'], list):
                comm_latency.append(np.mean(data['communication_latency']))
            else:
                comm_latency.append(data['communication_latency'])
        else:
            comm_latency.append(0)
    
    x = np.arange(len(experiments))
    width = 0.35
    
    ax.bar(x - width/2, training_latency, width, label='Training Latency', alpha=0.8)
    ax.bar(x + width/2, comm_latency, width, label='Communication Latency', alpha=0.8)
    
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Latency (seconds)')
    ax.set_title('Training vs Communication Latency')
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'latency_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_report(results, output_dir):
    """Create a summary report of all experiments"""
    report_path = os.path.join(output_dir, 'experiment_summary.txt')
    
    with open(report_path, 'w') as f:
        f.write("Edge-Aware Federated Learning - Experiment Summary\\n")
        f.write("="*60 + "\\n\\n")
        
        for exp_name, data in results.items():
            f.write(f"Experiment: {exp_name}\\n")
            f.write("-" * 30 + "\\n")
            
            # Extract key metrics
            if 'final_accuracy' in data:
                f.write(f"Final Accuracy: {data['final_accuracy']:.4f}\\n")
            
            if 'total_rounds' in data:
                f.write(f"Total Rounds: {data['total_rounds']}\\n")
            
            if 'total_time' in data:
                f.write(f"Total Time: {data['total_time']:.2f} seconds\\n")
            
            if 'avg_communication_overhead' in data:
                f.write(f"Avg Communication Overhead: {data['avg_communication_overhead']:.2f} MB\\n")
            
            if 'avg_energy_consumption' in data:
                f.write(f"Avg Energy Consumption: {data['avg_energy_consumption']:.2f} J\\n")
            
            f.write("\\n")
        
        f.write("\\nPlots generated:\\n")
        f.write("- training_curves.png\\n")
        f.write("- communication_overhead.png\\n")
        f.write("- energy_consumption.png\\n")
        f.write("- latency_analysis.png\\n")

def main():
    parser = argparse.ArgumentParser(description="Visualize FL experiment results")
    parser.add_argument("--results-dir", default="results", help="Directory containing experiment results")
    parser.add_argument("--output-dir", default="visualizations", help="Directory to save plots")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    print(f"Loading results from {args.results_dir}...")
    results = load_experiment_results(args.results_dir)
    
    if not results:
        print("No experiment results found!")
        return
    
    print(f"Found {len(results)} experiments")
    
    # Generate plots
    print("Generating visualizations...")
    plot_training_curves(results, args.output_dir)
    plot_communication_overhead(results, args.output_dir)
    plot_energy_consumption(results, args.output_dir)
    plot_latency_analysis(results, args.output_dir)
    
    # Create summary report
    create_summary_report(results, args.output_dir)
    
    print(f"SUCCESS: Visualizations saved to {args.output_dir}")
    print("SUMMARY: Summary report: experiment_summary.txt")

if __name__ == "__main__":
    main()
