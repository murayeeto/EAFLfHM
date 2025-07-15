#!/usr/bin/env python3
"""
Script to run federated learning experiments with different configurations
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from main import main
import argparse

def create_experiment_configs():
    """Create different experiment configurations"""
    experiments = [
        {
            "name": "small_scale",
            "description": "Small scale experiment with 5 clients, 10 rounds",
            "args": ["--num-clients", "5", "--num-rounds", "10", "--output-dir", "results/small_scale"]
        },
        {
            "name": "medium_scale", 
            "description": "Medium scale experiment with 20 clients, 50 rounds",
            "args": ["--num-clients", "20", "--num-rounds", "50", "--output-dir", "results/medium_scale"]
        },
        {
            "name": "large_scale",
            "description": "Large scale experiment with 100 clients, 100 rounds", 
            "args": ["--num-clients", "100", "--num-rounds", "100", "--output-dir", "results/large_scale"]
        },
        {
            "name": "energy_test",
            "description": "Energy consumption focused test",
            "args": ["--num-clients", "10", "--num-rounds", "20", "--output-dir", "results/energy_test"]
        }
    ]
    return experiments

async def run_experiment(experiment):
    """Run a single experiment"""
    print(f"\\n{'='*50}")
    print(f"Running experiment: {experiment['name']}")
    print(f"Description: {experiment['description']}")
    print(f"{'='*50}")
    
    # Create output directory
    output_dir = None
    for i, arg in enumerate(experiment['args']):
        if arg == "--output-dir" and i + 1 < len(experiment['args']):
            output_dir = experiment['args'][i + 1]
            break
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Temporarily replace sys.argv
    original_argv = sys.argv.copy()
    sys.argv = ["main.py"] + experiment['args']
    
    try:
        await main()
        print(f"✅ Experiment {experiment['name']} completed successfully")
    except Exception as e:
        print(f"❌ Experiment {experiment['name']} failed: {str(e)}")
    finally:
        sys.argv = original_argv

async def run_all_experiments():
    """Run all predefined experiments"""
    experiments = create_experiment_configs()
    
    print("🚀 Starting Edge-Aware FL Experiments")
    print(f"Total experiments to run: {len(experiments)}")
    
    for i, experiment in enumerate(experiments, 1):
        print(f"\\n📊 Progress: {i}/{len(experiments)}")
        await run_experiment(experiment)
        
        # Add delay between experiments
        if i < len(experiments):
            print("⏳ Waiting 30 seconds before next experiment...")
            await asyncio.sleep(30)
    
    print("\\n🎉 All experiments completed!")
    print("📁 Results saved in the 'results/' directory")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FL experiments")
    parser.add_argument("--experiment", choices=["all", "small", "medium", "large", "energy"], 
                       default="all", help="Which experiment(s) to run")
    
    args = parser.parse_args()
    
    if args.experiment == "all":
        asyncio.run(run_all_experiments())
    else:
        experiments = create_experiment_configs()
        experiment_map = {
            "small": experiments[0],
            "medium": experiments[1], 
            "large": experiments[2],
            "energy": experiments[3]
        }
        
        if args.experiment in experiment_map:
            asyncio.run(run_experiment(experiment_map[args.experiment]))
        else:
            print(f"Unknown experiment: {args.experiment}")
