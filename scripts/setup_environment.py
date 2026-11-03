#!/usr/bin/env python3
"""
Environment Setup Script

Sets up the research environment with all required dependencies
and configurations for RTX 5090 optimized research.
"""

import os
import sys
import subprocess
import platform
import json
import yaml
from pathlib import Path
from typing import List, Dict, Optional
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


class EnvironmentSetup:
    """Comprehensive environment setup for research."""
    
    def __init__(self, config_path: str = "./configs/lerna_research_2026.yaml"):
        self.config_path = Path(config_path)
        self.project_root = Path(__file__).parent.parent
        self.requirements_dir = self.project_root / "requirements"
        
        # Load configuration
        self.config = self._load_config()
        
        # System information
        self.system_info = self._get_system_info()
        
        logger.info(f"Environment Setup for LERNA Research")
        logger.info(f"   Project root: {self.project_root}")
        logger.info(f"   System: {self.system_info['system']} {self.system_info['release']}")
        logger.info(f"   Python: {self.system_info['python_version']}")
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            logger.warning(f"Configuration file not found: {self.config_path}")
            return {}
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _get_system_info(self) -> Dict:
        """Get system information."""
        return {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "cuda_available": self._check_cuda_available(),
        }
    
    def _check_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def create_directory_structure(self):
        """Create project directory structure."""
        logger.info("\n📁 Creating directory structure...")
        
        directories = [
            "configs",
            "lerna/utils",
            "lerna/callbacks",
            "scripts",
            "experiments/runs",
            "experiments/analysis",
            "experiments/artifacts",
            "requirements",
            "data/raw",
            "data/processed",
            "notebooks",
            "reports",
            "logs",
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"   Created: {directory}")
        
        # Create README files
        self._create_readme_files()
        
        logger.info("✅ Directory structure created")
    
    def _create_readme_files(self):
        """Create README files for directories."""
        readme_content = {
            "experiments": """
# Experiments Directory

This directory contains all experiment outputs.

## Structure:
- `runs/` - Individual experiment runs with results
- `analysis/` - Aggregated analysis and reports
- `artifacts/` - Saved models and checkpoints

## Naming Convention:
- Experiments: `{model}_{task}_s{seed}_lr{lr:.1e}`
- Analysis: `analysis_{timestamp}_{purpose}`
- Artifacts: `artifact_{model}_{task}_{version}`
""",
            "data": """
# Data Directory

This directory contains datasets and processed data.

## Structure:
- `raw/` - Original datasets (not version controlled)
- `processed/` - Preprocessed data for experiments

## Notes:
- Large datasets should not be committed to git
- Use `.gitignore` to exclude raw data
- Document data preprocessing steps
""",
            "notebooks": """
# Notebooks Directory

Jupyter notebooks for exploratory analysis and visualization.

## Structure:
- `exploratory/` - Initial data exploration
- `analysis/` - Result analysis and visualization
- `prototypes/` - Method prototyping

## Best Practices:
- Clear cell numbering and headings
- Document assumptions and findings
- Convert to scripts for production use
""",
        }
        
        for dir_name, content in readme_content.items():
            readme_path = self.project_root / dir_name / "README.md"
            with open(readme_path, 'w') as f:
                f.write(content.strip())
    
    def install_dependencies(self, requirements_file: str = "requirements.txt"):
        """Install Python dependencies."""
        logger.info(f"\n📦 Installing dependencies from {requirements_file}...")
        
        req_path = self.requirements_dir / requirements_file
        if not req_path.exists():
            logger.error(f"Requirements file not found: {req_path}")
            return False
        
        # Install using pip
        cmd = [sys.executable, "-m", "pip", "install", "-r", str(req_path)]
        
        try:
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"✅ Dependencies installed successfully")
            logger.debug(f"Output: {result.stdout[:500]}...")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install dependencies:")
            logger.error(f"Error: {e.stderr}")
            return False
    
    def install_research_dependencies(self):
        """Install research-specific dependencies."""
        logger.info("\n🔬 Installing research-specific dependencies...")
        
        research_req = self.requirements_dir / "requirements_research.txt"
        if research_req.exists():
            return self.install_dependencies("requirements_research.txt")
        else:
            logger.warning(f"Research requirements file not found: {research_req}")
            return False
    
    def setup_wandb(self):
        """Setup Weights & Biases for experiment tracking."""
        logger.info("\n📊 Setting up Weights & Biases...")
        
        # Check if wandb is installed
        try:
            import wandb
            logger.info(f"   W&B version: {wandb.__version__}")
        except ImportError:
            logger.warning("   W&B not installed. Installing...")
            cmd = [sys.executable, "-m", "pip", "install", "wandb"]
            subprocess.run(cmd, check=False)
        
        # Check if wandb is configured
        wandb_dir = Path.home() / ".netrc"
        if wandb_dir.exists():
            logger.info("   W&B configuration found")
        else:
            logger.warning("   W&B not configured. Please run: wandb login")
            logger.info("   Get your API key from: https://wandb.ai/authorize")
        
        # Create example wandb config
        wandb_config = {
            "entity": self.config.get("wandb", {}).get("entity", "your-entity"),
            "project": self.config.get("wandb", {}).get("project", "lerna-research"),
            "tags": ["research", "efficiency", "nlp"],
            "notes": "LERNA: Energy-Efficient LLM Fine-Tuning",
        }
        
        config_path = self.project_root / "configs" / "wandb_config.json"
        with open(config_path, 'w') as f:
            json.dump(wandb_config, f, indent=2)
        
        logger.info(f"   Example W&B config saved: {config_path}")
        logger.info("✅ W&B setup complete")
    
    def setup_gpu_environment(self):
        """Setup GPU-optimized environment for RTX 5090."""
        logger.info("\n⚡ Setting up GPU-optimized environment...")
        
        # Check CUDA availability
        if not self.system_info["cuda_available"]:
            logger.warning("   CUDA not available. GPU optimizations may not work.")
            return
        
        # Check PyTorch CUDA version
        try:
            import torch
            cuda_version = torch.version.cuda
            logger.info(f"   PyTorch CUDA version: {cuda_version}")
            
            # Check GPU
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"   GPU: {gpu_name}")
            logger.info(f"   GPU Memory: {gpu_memory:.1f} GB")
            
            # Test tensor operations
            test_tensor = torch.randn(1000, 1000).cuda()
            result = test_tensor @ test_tensor.T
            logger.info(f"   GPU test passed: {result.shape} matrix multiplication")
            
        except Exception as e:
            logger.error(f"   GPU test failed: {e}")
        
        # Setup Flash Attention if available
        self._setup_flash_attention()
        
        # Setup Torch Compile
        self._setup_torch_compile()
        
        logger.info("✅ GPU environment setup complete")
    
    def _setup_flash_attention(self):
        """Setup Flash Attention 2."""
        logger.info("   Checking Flash Attention 2...")
        
        try:
            # Try to import flash attention
            import flash_attn
            logger.info(f"   Flash Attention 2 available: {flash_attn.__version__}")
            
            # Test flash attention
            import torch
            from flash_attn import flash_attn_func
            
            # Create test tensors
            batch_size, seq_len, n_heads, head_dim = 2, 512, 12, 64
            q = torch.randn(batch_size, seq_len, n_heads, head_dim).cuda().half()
            k = torch.randn(batch_size, seq_len, n_heads, head_dim).cuda().half()
            v = torch.randn(batch_size, seq_len, n_heads, head_dim).cuda().half()
            
            # Run flash attention
            output = flash_attn_func(q, k, v)
            logger.info(f"   Flash Attention test passed: {output.shape}")
            
        except ImportError:
            logger.warning("   Flash Attention 2 not installed.")
            logger.info("   Install with: pip install flash-attn --no-build-isolation")
        
        except Exception as e:
            logger.error(f"   Flash Attention test failed: {e}")
    
    def _setup_torch_compile(self):
        """Setup Torch Compile optimizations."""
        logger.info("   Checking Torch Compile...")
        
        try:
            import torch
            
            # Check if torch.compile is available
            if hasattr(torch, 'compile'):
                logger.info(f"   Torch Compile available (PyTorch {torch.__version__})")
                
                # Test compile
                test_model = torch.nn.Linear(100, 100).cuda()
                compiled_model = torch.compile(test_model, mode="max-autotune")
                
                # Test inference
                x = torch.randn(10, 100).cuda()
                output = compiled_model(x)
                logger.info(f"   Torch Compile test passed: {output.shape}")
            else:
                logger.warning("   Torch Compile not available (requires PyTorch 2.0+)")
                
        except Exception as e:
            logger.error(f"   Torch Compile test failed: {e}")
    
    def create_environment_config(self):
        """Create environment configuration file."""
        logger.info("\n⚙️  Creating environment configuration...")
        
        env_config = {
            "timestamp": self._get_timestamp(),
            "project": "LERNA",
            "version": "research-2.0",
            "system": self.system_info,
            "python_dependencies": self._get_installed_packages(),
            "environment_variables": self._get_environment_vars(),
            "gpu_configuration": self._get_gpu_config(),
            "paths": {
                "project_root": str(self.project_root),
                "configs": str(self.project_root / "configs"),
                "experiments": str(self.project_root / "experiments"),
                "data": str(self.project_root / "data"),
                "logs": str(self.project_root / "logs"),
            },
            "setup_notes": [
                "Environment configured for RTX 5090 optimized research",
                "Includes Flash Attention 2 and Torch Compile optimizations",
                "W&B integration for experiment tracking",
                "Statistical analysis framework included",
            ],
        }
        
        config_path = self.project_root / "configs" / "environment_config.json"
        with open(config_path, 'w') as f:
            json.dump(env_config, f, indent=2, default=str)
        
        logger.info(f"✅ Environment config saved: {config_path}")
        
        return env_config
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _get_installed_packages(self) -> Dict[str, str]:
        """Get installed Python packages."""
        try:
            import pkg_resources
            packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
            return packages
        except:
            return {}
    
    def _get_environment_vars(self) -> Dict[str, str]:
        """Get relevant environment variables."""
        env_vars = {
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
            "WANDB_API_KEY": "SET" if os.environ.get("WANDB_API_KEY") else "NOT_SET",
            "TORCH_CUDA_ARCH_LIST": os.environ.get("TORCH_CUDA_ARCH_LIST", ""),
        }
        return env_vars
    
    def _get_gpu_config(self) -> Dict:
        """Get GPU configuration."""
        try:
            import torch
            if torch.cuda.is_available():
                return {
                    "device_count": torch.cuda.device_count(),
                    "current_device": torch.cuda.current_device(),
                    "device_name": torch.cuda.get_device_name(0),
                    "device_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
                    "cuda_version": torch.version.cuda,
                    "cudnn_version": torch.backends.cudnn.version() if hasattr(torch.backends.cudnn, 'version') else "unknown",
                }
        except:
            pass
        return {"available": False}
    
    def validate_installation(self):
        """Validate the installation."""
        logger.info("\n🔍 Validating installation...")
        
        validation_results = {
            "directories": self._validate_directories(),
            "dependencies": self._validate_dependencies(),
            "gpu": self._validate_gpu(),
            "wandb": self._validate_wandb(),
        }
        
        # Print validation summary
        logger.info("\n📋 Validation Summary:")
        for component, result in validation_results.items():
            status = "✅" if result["valid"] else "❌"
            logger.info(f"   {component}: {status} {result.get('message', '')}")
        
        # Save validation report
        self._save_validation_report(validation_results)
        
        return validation_results
    
    def _validate_directories(self) -> Dict:
        """Validate directory structure."""
        required_dirs = [
            "configs",
            "lerna/utils",
            "lerna/callbacks",
            "scripts",
            "experiments/runs",
            "requirements",
        ]
        
        missing_dirs = []
        for dir_path in required_dirs:
            if not (self.project_root / dir_path).exists():
                missing_dirs.append(dir_path)
        
        valid = len(missing_dirs) == 0
        message = f"Missing directories: {missing_dirs}" if missing_dirs else "All directories present"
        
        return {"valid": valid, "message": message, "missing": missing_dirs}
    
    def _validate_dependencies(self) -> Dict:
        """Validate Python dependencies."""
        required_packages = [
            "torch",
            "transformers",
            "datasets",
            "numpy",
            "pandas",
            "scipy",
            "matplotlib",
            "seaborn",
            "wandb",
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        valid = len(missing_packages) == 0
        message = f"Missing packages: {missing_packages}" if missing_packages else "All packages installed"
        
        return {"valid": valid, "message": message, "missing": missing_packages}
    
    def _validate_gpu(self) -> Dict:
        """Validate GPU setup."""
        try:
            import torch
            if torch.cuda.is_available():
                # Test GPU operations
                x = torch.randn(1000, 1000).cuda()
                y = torch.randn(1000, 1000).cuda()
                z = x @ y  # Matrix multiplication
                
                return {
                    "valid": True,
                    "message": f"GPU available: {torch.cuda.get_device_name(0)}",
                    "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
                }
            else:
                return {
                    "valid": False,
                    "message": "CUDA not available",
                }
        except Exception as e:
            return {
                "valid": False,
                "message": f"GPU test failed: {e}",
            }
    
    def _validate_wandb(self) -> Dict:
        """Validate W&B setup."""
        try:
            import wandb
            # Try to get version
            version = wandb.__version__
            
            # Check if logged in (by trying to create a dummy run)
            try:
                run = wandb.init(mode="disabled")
                run.finish()
                logged_in = True
            except:
                logged_in = False
            
            return {
                "valid": True,
                "message": f"W&B version {version}" + (" (logged in)" if logged_in else " (not logged in)"),
                "logged_in": logged_in,
            }
        except ImportError:
            return {
                "valid": False,
                "message": "W&B not installed",
            }
        except Exception as e:
            return {
                "valid": False,
                "message": f"W&B validation failed: {e}",
            }
    
    def _save_validation_report(self, results: Dict):
        """Save validation report."""
        report = {
            "timestamp": self._get_timestamp(),
            "project": "LERNA",
            "validation_results": results,
            "overall_status": all(r["valid"] for r in results.values()),
            "next_steps": self._get_next_steps(results),
        }
        
        report_path = self.project_root / "logs" / "setup_validation.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📁 Validation report saved: {report_path}")
    
    def _get_next_steps(self, results: Dict) -> List[str]:
        """Get next steps based on validation results."""
        next_steps = []
        
        if not results["directories"]["valid"]:
            next_steps.append("Run setup with --create-dirs to create missing directories")
        
        if not results["dependencies"]["valid"]:
            next_steps.append("Install missing dependencies from requirements/ directory")
        
        if not results["gpu"]["valid"]:
            next_steps.append("Check CUDA installation and GPU drivers")
        
        if not results["wandb"]["valid"] or not results["wandb"].get("logged_in", False):
            next_steps.append("Run: wandb login to set up experiment tracking")
        
        if not next_steps:
            next_steps.append("Setup complete! Run: python scripts/run_research_sweep.py to start experiments")
        
        return next_steps
    
    def run_complete_setup(self):
        """Run complete environment setup."""
        logger.info(f"\n{'='*60}")
        logger.info("COMPLETE ENVIRONMENT SETUP")
        logger.info(f"{'='*60}")
        
        # 1. Create directory structure
        self.create_directory_structure()
        
        # 2. Install dependencies
        self.install_dependencies()
        self.install_research_dependencies()
        
        # 3. Setup W&B
        self.setup_wandb()
        
        # 4. Setup GPU environment
        self.setup_gpu_environment()
        
        # 5. Create environment config
        self.create_environment_config()
        
        # 6. Validate installation
        validation = self.validate_installation()
        
        logger.info(f"\n{'='*60}")
        logger.info("SETUP COMPLETE")
        logger.info(f"{'='*60}")
        
        # Print next steps
        if validation.get("overall_status", False):
            logger.info("\n🎉 Environment is ready for research!")
            logger.info("\nNext steps:")
            logger.info("1. Configure your experiment in configs/lerna_research_2026.yaml")
            logger.info("2. Run: python scripts/run_research_sweep.py --config configs/lerna_research_2026.yaml")
            logger.info("3. Monitor experiments: wandb.ai")
        else:
            logger.info("\n⚠️  Setup completed with issues:")
            for step in validation.get("next_steps", []):
                logger.info(f"  • {step}")
        
        logger.info(f"\nFor help, check: {self.project_root}/README.md")
        logger.info(f"{'='*60}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Setup research environment")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/lerna_research_2026.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--create-dirs",
        action="store_true",
        help="Create directory structure only",
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install dependencies only",
    )
    parser.add_argument(
        "--setup-wandb",
        action="store_true",
        help="Setup Weights & Biases only",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate installation only",
    )
    parser.add_argument(
        "--complete",
        action="store_true",
        help="Run complete setup (default)",
    )
    
    args = parser.parse_args()
    
    # Initialize setup
    setup = EnvironmentSetup(config_path=args.config)
    
    # Run requested actions
    if args.create_dirs:
        setup.create_directory_structure()
    elif args.install_deps:
        setup.install_dependencies()
        setup.install_research_dependencies()
    elif args.setup_wandb:
        setup.setup_wandb()
    elif args.validate:
        setup.validate_installation()
    else:
        # Default: complete setup
        setup.run_complete_setup()


if __name__ == "__main__":
    main()