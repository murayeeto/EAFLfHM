"""
Edge-Aware Federated Learning System Launcher
Interactive command-line interface for the FL system
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def print_banner():
    """Print system banner"""
    banner = """
╔════════════════════════════════════════════════════════════════╗
║                Edge-Aware Federated Learning                   ║  
║              Real-Time Health Monitoring System               ║
╚════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_menu():
    """Print main menu options"""
    menu = """
🎯 What would you like to do?

1️⃣  Quick Demo (5 minutes)           - Interactive system demonstration
2️⃣  Run Small Experiment (2 min)     - 5 clients, 10 rounds  
3️⃣  Run Medium Experiment (10 min)   - 20 clients, 50 rounds
4️⃣  Run Large Experiment (30 min)    - 100 clients, 100 rounds
5️⃣  Custom Experiment                - Specify your own parameters
6️⃣  Visualize Results                - Generate plots from previous runs
7️⃣  Run System Tests                 - Verify system functionality
8️⃣  Setup System                     - Install dependencies and configure
9️⃣  Docker Deployment                - Run containerized system
🔧  Advanced Options                  - Individual components and tools
❓  Help & Documentation             - Usage guides and examples
🚪  Exit

Enter your choice (1-9, or 'help'/'exit'): """
    
    return input(menu).strip().lower()

def print_advanced_menu():
    """Print advanced options menu"""
    menu = """
🔧 Advanced Options:

a) Start Edge Server              - Run individual edge server
b) Start Wearable Client          - Simulate single device  
c) FL Coordinator Only            - Run coordination service
d) Network Simulation             - Test network conditions
e) Energy Analysis                - Battery consumption study
f) Communication Overhead Test    - Network usage analysis
g) Model Architecture Test        - Validate ML models
h) Configuration Editor           - Modify system settings
i) Performance Profiling         - System performance analysis
j) Back to Main Menu

Enter your choice: """
    
    return input(menu).strip().lower()

async def run_quick_demo():
    """Run the quick demo"""
    print("🚀 Starting Quick Demo...")
    try:
        from examples.simple_demo import main as demo_main
        await demo_main()
    except ImportError:
        print("❌ Demo module not found. Please ensure examples/simple_demo.py exists.")
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")

async def run_experiment(size):
    """Run predefined experiments"""
    experiments = {
        "small": ("small", "5 clients, 10 rounds"),
        "medium": ("medium", "20 clients, 50 rounds"), 
        "large": ("large", "100 clients, 100 rounds")
    }
    
    if size in experiments:
        exp_type, description = experiments[size]
        print(f"🔬 Running {size} experiment: {description}")
        
        # Import main directly and run with modified sys.argv
        import sys
        original_argv = sys.argv.copy()
        
        experiments_config = {
            "small": ["--num-clients", "5", "--num-rounds", "10", "--output-dir", "results/launcher_small"],
            "medium": ["--num-clients", "20", "--num-rounds", "50", "--output-dir", "results/launcher_medium"],
            "large": ["--num-clients", "100", "--num-rounds", "100", "--output-dir", "results/launcher_large"]
        }
        
        sys.argv = ["main.py"] + experiments_config[size]
        
        try:
            from main import main
            await main()
            print(f"✅ {size.capitalize()} experiment completed successfully!")
        except ImportError:
            print("❌ Main module not found. Please ensure main.py exists in the project directory.")
        except Exception as e:
            print(f"❌ Experiment failed: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            sys.argv = original_argv
    else:
        print("❌ Invalid experiment size")

async def run_custom_experiment():
    """Run custom experiment with user parameters"""
    print("🎛️ Custom Experiment Setup")
    print("=" * 30)
    
    try:
        num_clients = int(input("Number of clients (default 10): ") or "10")
        num_rounds = int(input("Number of FL rounds (default 20): ") or "20")
        output_dir = input("Output directory (default results/custom): ") or "results/custom"
        
        print(f"\\n🔬 Running experiment: {num_clients} clients, {num_rounds} rounds")
        
        # Import and run main with custom args
        import sys
        original_argv = sys.argv.copy()
        sys.argv = [
            "main.py",
            "--num-clients", str(num_clients),
            "--num-rounds", str(num_rounds), 
            "--output-dir", output_dir
        ]
        
        try:
            from main import main
            await main()
            print("✅ Custom experiment completed!")
        except ImportError:
            print("❌ Main module not found. Please ensure main.py exists in the project directory.")
        finally:
            sys.argv = original_argv
        
    except ValueError:
        print("❌ Invalid input. Please enter numeric values.")
    except Exception as e:
        print(f"❌ Experiment failed: {str(e)}")

def visualize_results():
    """Run visualization script"""
    print("Generating visualizations...")
    script_path = project_root / "scripts" / "visualize_results.py"
    
    if not script_path.exists():
        print("❌ Visualization script not found. Please ensure scripts/visualize_results.py exists.")
        return
        
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, str(script_path)
        ], capture_output=True, text=True, encoding='utf-8', errors='replace')
        
        if result.returncode == 0:
            print("SUCCESS: Visualizations generated successfully!")
            print("📊 Generated plots:")
            print("  • Training curves (accuracy/loss)")
            print("  • Communication overhead analysis")
            print("  • Energy consumption metrics")
            print("  • Latency analysis")
            print("  • Confusion matrix heatmap (false positives/negatives)")
            print("\\nCheck the 'visualizations' directory for all plots")
            if result.stdout:
                print(result.stdout)
        else:
            print(f"ERROR: Visualization failed")
            if result.stderr:
                print(f"Error details: {result.stderr}")
    except Exception as e:
        print(f"ERROR: Error running visualization: {str(e)}")

def run_tests():
    """Run system tests"""
    print("🧪 Running system tests...")
    test_path = project_root / "tests" / "test_system.py"
    
    if not test_path.exists():
        print("❌ Test script not found. Please ensure tests/test_system.py exists.")
        return
        
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, str(test_path)
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.returncode == 0:
            print("✅ All tests passed!")
        else:
            print(f"❌ Some tests failed: {result.stderr}")
    except Exception as e:
        print(f"❌ Error running tests: {str(e)}")

def setup_system():
    """Run setup script"""
    print("⚙️ Setting up system...")
    setup_path = project_root / "setup.py"
    
    if not setup_path.exists():
        print("❌ Setup script not found. Please ensure setup.py exists.")
        return
        
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, str(setup_path)
        ])
        
        if result.returncode == 0:
            print("✅ Setup completed successfully!")
        else:
            print("❌ Setup failed")
    except Exception as e:
        print(f"❌ Error during setup: {str(e)}")

def docker_deployment():
    """Docker deployment options"""
    print("🐳 Docker Deployment Options:")
    print("1. Full system (docker-compose)")
    print("2. Edge server only")
    print("3. View docker status")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        print("🚀 Starting full Docker deployment...")
        os.system("cd docker && docker-compose up")
    elif choice == "2":
        print("🏢 Starting edge server container...")
        os.system("docker run -p 8080:8080 edge-fl:latest")
    elif choice == "3":
        print("📊 Docker status:")
        os.system("docker ps")
    else:
        print("❌ Invalid choice")

async def advanced_options():
    """Handle advanced options"""
    while True:
        choice = print_advanced_menu()
        
        if choice in ['j', 'back', 'main']:
            break
        elif choice == 'a':
            print("🏢 Starting edge server...")
            # Add edge server startup logic
            print("Edge server would start here (not implemented in demo)")
        elif choice == 'b':
            print("👤 Starting wearable client...")
            # Add client startup logic  
            print("Wearable client would start here (not implemented in demo)")
        elif choice == 'h':
            print("⚙️ Configuration editor not implemented yet")
        else:
            print("❌ Invalid choice")

def show_help():
    """Show help and documentation"""
    help_text = """
📖 Help & Documentation

📋 Quick Commands:
   python main.py --help                    # Command line help
   python examples/simple_demo.py           # Interactive demo
   python scripts/run_experiments.py        # Automated experiments
   python scripts/visualize_results.py      # Generate plots

📁 Important Files:
   README.md                                # Full documentation
   QUICKSTART.md                           # 5-minute guide
   config/config.py                        # System configuration
   requirements.txt                        # Dependencies

🔗 Key Directories:
   results/                                # Experiment outputs
   visualizations/                         # Generated plots
   logs/                                   # System logs
   examples/                               # Usage examples

🧪 Testing:
   python tests/test_system.py             # Run all tests
   python setup.py                         # Verify installation

🐳 Docker:
   cd docker && docker-compose up          # Full system
   
❓ Need more help? Check README.md for comprehensive documentation.
"""
    print(help_text)

async def main():
    """Main launcher function"""
    print_banner()
    
    while True:
        try:
            choice = print_menu()
            
            if choice in ['9', 'exit', 'quit', 'q']:
                print("👋 Goodbye! Thanks for using Edge-Aware FL!")
                break
            elif choice == '1':
                await run_quick_demo()
            elif choice == '2':
                await run_experiment("small")
            elif choice == '3':
                await run_experiment("medium")
            elif choice == '4':
                await run_experiment("large")
            elif choice == '5':
                await run_custom_experiment()
            elif choice == '6':
                visualize_results()
            elif choice == '7':
                run_tests()
            elif choice == '8':
                setup_system()
            elif choice in ['9', 'docker']:
                docker_deployment()
            elif choice in ['advanced', 'a']:
                await advanced_options()
            elif choice in ['help', 'h', '?']:
                show_help()
            else:
                print("❌ Invalid choice. Please try again.")
                
            input("\\n⏸️  Press Enter to continue...")
            
        except KeyboardInterrupt:
            print("\\n\\n⏹️  Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\\n❌ Error: {str(e)}")
            input("⏸️  Press Enter to continue...")

if __name__ == "__main__":
    asyncio.run(main())
    asyncio.run(main())
