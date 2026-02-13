#!/usr/bin/env python3
"""
Installation Validation Script

Validates that all components are properly installed and configured
for the research project.
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import argparse
import logging
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


class InstallationValidator:
    """Comprehensive installation validator."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.results = {
            "basic_checks": {},
            "package_checks": {},
            "gpu_checks": {},
            "project_checks": {},
            "integration_checks": {},
        }
    
    def run_all_checks(self, quick: bool = False) -> Dict:
        """Run all validation checks."""
        logger.info(f"\n{'='*60}")
        logger.info("COMPREHENSIVE INSTALLATION VALIDATION")
        logger.info(f"{'='*60}")
        
        # Run checks
        self.results["basic_checks"] = self.check_basic_requirements()
        self.results["package_checks"] = self.check_packages()
        
        if not quick:
            self.results["gpu_checks"] = self.check_gpu_environment()
        else:
            self.results["gpu_checks"] = {"skipped": {"passed": True}}
            logger.info("\n⚡ Skipping GPU checks (--quick mode)")
        
        self.results["project_checks"] = self.check_project_structure()
        self.results["integration_checks"] = self.check_integrations()
        
        # Generate summary
        summary = self.generate_summary()
        
        # Save results
        self.save_results(summary)
        
        return summary
    
    def _get_package_import_name(self, package_name: str) -> str:
        """Get the actual import name for a package."""
        import_name_map = {
            "scikit-learn": "sklearn",
            "pyyaml": "yaml",
            # Add other mappings as needed
        }
        return import_name_map.get(package_name, package_name)
    
    def check_basic_requirements(self) -> Dict:
        """Check basic system requirements."""
        logger.info("\n🔍 Checking basic requirements...")
        
        checks = {}
        
        # Python version
        python_version = sys.version_info
        checks["python_version"] = {
            "required": "3.8+",
            "actual": f"{python_version.major}.{python_version.minor}.{python_version.micro}",
            "passed": python_version >= (3, 8),
        }
        
        # Operating system
        import platform
        checks["operating_system"] = {
            "system": platform.system(),
            "release": platform.release(),
            "passed": True,  # All major OSes supported
        }
        
        # Memory
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        checks["system_memory"] = {
            "required": "16 GB",
            "actual": f"{memory_gb:.1f} GB",
            "passed": memory_gb >= 16,
        }
        
        # Disk space - improved check with fallbacks
        try:
            # Try multiple paths for disk space check
            paths_to_check = [
                str(self.project_root),
                str(Path.home()),
                "/tmp",
                "/"  # Root directory
            ]
            
            for path in paths_to_check:
                try:
                    disk_usage = psutil.disk_usage(path)
                    free_gb = disk_usage.free / (1024**3)
                    
                    checks["disk_space"] = {
                        "required": "50 GB",
                        "actual": f"{free_gb:.1f} GB free at {path}",
                        "passed": free_gb >= 50,
                        "location": path,
                        "total_gb": f"{disk_usage.total / (1024**3):.1f} GB",
                        "used_gb": f"{disk_usage.used / (1024**3):.1f} GB",
                    }
                    
                    # If we found a valid path with enough space, break
                    if free_gb >= 50:
                        break
                        
                except Exception as e:
                    if path == paths_to_check[-1]:  # Last path, raise the error
                        raise
                    continue
                    
        except Exception as e:
            checks["disk_space"] = {
                "required": "50 GB",
                "actual": f"Error checking disk space: {str(e)[:100]}",
                "passed": False,
                "warning": "Disk space check failed, but this may be a permission issue. Continuing...",
                "optional": True,  # Mark as optional for overall status
            }
        
        # Log results
        for check_name, check_result in checks.items():
            status = "✅" if check_result["passed"] else "❌"
            logger.info(f"   {check_name}: {status} {check_result.get('actual', '')}")
            if "warning" in check_result:
                logger.warning(f"     ⚠️  {check_result['warning']}")
        
        return checks
    
    def check_packages(self) -> Dict:
        """Check required Python packages."""
        logger.info("\n📦 Checking Python packages...")
        
        required_packages = {
            "torch": {"min_version": "2.0.0", "purpose": "Deep learning framework"},
            "transformers": {"min_version": "4.30.0", "purpose": "NLP models"},
            "datasets": {"min_version": "2.10.0", "purpose": "Dataset handling"},
            "numpy": {"min_version": "1.21.0", "purpose": "Numerical computations"},
            "pandas": {"min_version": "1.3.0", "purpose": "Data analysis"},
            "scipy": {"min_version": "1.7.0", "purpose": "Scientific computing"},
            "matplotlib": {"min_version": "3.5.0", "purpose": "Plotting"},
            "seaborn": {"min_version": "0.11.0", "purpose": "Statistical visualizations"},
            "wandb": {"min_version": "0.15.0", "purpose": "Experiment tracking"},
            "scikit-learn": {"min_version": "1.0.0", "purpose": "Machine learning utilities"},
            "tqdm": {"min_version": "4.62.0", "purpose": "Progress bars"},
            "pyyaml": {"min_version": "6.0", "purpose": "YAML configuration"},
            "psutil": {"min_version": "5.9.0", "purpose": "System monitoring"},
            "packaging": {"min_version": "23.0", "purpose": "Version parsing"},
        }
        
        checks = {}
        
        for package_name, requirements in required_packages.items():
            try:
                # Use the mapped import name if available, otherwise use the package name
                import_name = self._get_package_import_name(package_name)
                module = __import__(import_name)
                
                # Get version (handle special cases)
                version = getattr(module, "__version__", "unknown")
                
                # For scikit-learn (sklearn), check for version in different attribute
                if package_name == "scikit-learn" and version == "unknown":
                    # sklearn might store version in _version.VERSION
                    try:
                        from sklearn import __version__ as sklearn_version
                        version = sklearn_version
                    except ImportError:
                        version = "unknown"
                
                # Check version
                from packaging import version as pkg_version
                min_ver = requirements["min_version"]
                
                # Handle 'unknown' version case
                if version == "unknown":
                    passed = False
                    logger.warning(f"   {package_name}: ⚠️  version unknown")
                else:
                    passed = pkg_version.parse(version) >= pkg_version.parse(min_ver)
                
                checks[package_name] = {
                    "required": min_ver,
                    "actual": version,
                    "passed": passed,
                    "purpose": requirements["purpose"],
                }
                
                status = "✅" if passed else "❌"
                logger.info(f"   {package_name}: {status} {version} (requires {min_ver})")
                
            except ImportError as e:
                checks[package_name] = {
                    "required": requirements["min_version"],
                    "actual": "not installed",
                    "passed": False,
                    "purpose": requirements["purpose"],
                    "error": str(e),
                }
                logger.info(f"   {package_name}: ❌ not installed")
        
        return checks
    
    def check_gpu_environment(self) -> Dict:
        """Check GPU environment and optimizations."""
        logger.info("\n⚡ Checking GPU environment...")
        
        checks = {}
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        checks["cuda_available"] = {
            "required": True,
            "actual": cuda_available,
            "passed": cuda_available,
        }
        
        if cuda_available:
            # GPU information
            gpu_count = torch.cuda.device_count()
            checks["gpu_count"] = {
                "count": gpu_count,
                "passed": gpu_count > 0,
            }
            
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            cuda_version = torch.version.cuda
            
            checks["gpu_info"] = {
                "name": gpu_name,
                "memory_gb": gpu_memory,
                "cuda_version": cuda_version,
                "passed": True,
            }
            
            # Check compute capability
            compute_capability = torch.cuda.get_device_capability(0)
            checks["compute_capability"] = {
                "required": "7.0+",
                "actual": f"{compute_capability[0]}.{compute_capability[1]}",
                "passed": compute_capability[0] >= 7,
            }
            
            # Test GPU operations (safer size)
            try:
                # Test matrix multiplication with safer size
                a = torch.randn(512, 512, device="cuda")
                b = torch.randn(512, 512, device="cuda")
                c = torch.matmul(a, b)
                
                checks["gpu_operations"] = {
                    "test": "Matrix multiplication (512x512)",
                    "passed": True,
                    "result_shape": list(c.shape),
                }
            except Exception as e:
                checks["gpu_operations"] = {
                    "test": "Matrix multiplication",
                    "passed": False,
                    "error": str(e),
                }
            
            # Check PyTorch SDPA (Scaled Dot Product Attention) - Built-in Flash Attention
            if hasattr(torch.backends.cuda, 'flash_sdp_enabled'):
                flash_enabled = torch.backends.cuda.flash_sdp_enabled()
                mem_efficient_enabled = torch.backends.cuda.mem_efficient_sdp_enabled()
                math_enabled = torch.backends.cuda.math_sdp_enabled()
                
                checks["pytorch_sdpa"] = {
                    "flash_available": flash_enabled,
                    "mem_efficient_available": mem_efficient_enabled,
                    "math_available": math_enabled,
                    "passed": flash_enabled or mem_efficient_enabled,
                    "note": "Built-in PyTorch optimized attention (replaces flash-attn package)",
                }
                
                if flash_enabled:
                    logger.info("   PyTorch Flash SDPA: ✅ Available (Flash Attention built-in)")
                elif mem_efficient_enabled:
                    logger.info("   PyTorch Memory Efficient SDPA: ✅ Available")
                else:
                    logger.info("   PyTorch SDPA: ⚠️ Only math backend available")
            else:
                checks["pytorch_sdpa"] = {
                    "passed": False,
                    "note": "PyTorch 2.0+ required for SDPA",
                }
                logger.info("   PyTorch SDPA: ❌ Not available")
            
            # Test SDPA functionality
            try:
                import torch.nn.functional as F
                # Create test tensors
                query = torch.randn(2, 4, 128, 64, device="cuda")
                key = torch.randn(2, 4, 128, 64, device="cuda")
                value = torch.randn(2, 4, 128, 64, device="cuda")
                
                output = F.scaled_dot_product_attention(query, key, value, is_causal=True)
                checks["sdpa_test"] = {
                    "test": "Scaled Dot Product Attention",
                    "passed": True,
                    "result_shape": list(output.shape),
                }
                logger.info("   SDPA Test: ✅ Works")
            except Exception as e:
                checks["sdpa_test"] = {
                    "test": "Scaled Dot Product Attention",
                    "passed": False,
                    "error": str(e)[:100],
                }
                logger.info(f"   SDPA Test: ❌ Failed: {str(e)[:50]}")
            
            # Check Torch Compile
            if hasattr(torch, 'compile'):
                checks["torch_compile"] = {
                    "available": True,
                    "passed": True,
                }
                logger.info("   Torch Compile: ✅ available")
            else:
                checks["torch_compile"] = {
                    "available": False,
                    "passed": False,
                    "note": "Requires PyTorch 2.0+",
                }
                logger.info("   Torch Compile: ⚠️  not available")
        
        # Log GPU info
        if cuda_available:
            logger.info(f"   GPU Count: ✅ {gpu_count}")
            logger.info(f"   GPU: ✅ {gpu_name} ({gpu_memory:.1f} GB)")
            logger.info(f"   CUDA: ✅ {cuda_version}")
        else:
            logger.info("   GPU: ❌ CUDA not available")
        
        return checks
    
    def check_project_structure(self) -> Dict:
        """Check project directory structure."""
        logger.info("\n📁 Checking project structure...")
        
        required_dirs = [
            "configs",
            "pillar0/utils",
            "pillar0/callbacks",
            "scripts",
            "experiments/runs",
            "experiments/analysis",
            "experiments/artifacts",
            "requirements",
        ]
        
        required_files = [
            "configs/pillar0_research_2026.yaml",
            "pillar0/__init__.py",
            "scripts/run_research_sweep.py",
            "requirements/requirements.txt",
            "README.md",
        ]
        
        checks = {"directories": {}, "files": {}}
        
        # Check directories
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            exists = full_path.exists() and full_path.is_dir()
            checks["directories"][dir_path] = {
                "exists": exists,
                "passed": exists,
            }
            status = "✅" if exists else "❌"
            logger.info(f"   Directory {dir_path}: {status}")
        
        # Check files
        for file_path in required_files:
            full_path = self.project_root / file_path
            exists = full_path.exists() and full_path.is_file()
            checks["files"][file_path] = {
                "exists": exists,
                "passed": exists,
            }
            status = "✅" if exists else "❌"
            logger.info(f"   File {file_path}: {status}")
        
        return checks
    
    def check_integrations(self) -> Dict:
        """Check external integrations."""
        logger.info("\n🔗 Checking external integrations...")
        
        checks = {}
        
        # Check Weights & Biases
        try:
            import wandb
            
            # Check if wandb is installed
            checks["wandb_installed"] = {
                "passed": True,
                "version": wandb.__version__,
            }
            
            # Better login check
            logged_in = wandb.api.api_key is not None
            
            checks["wandb_logged_in"] = {
                "passed": logged_in,
                "note": "Run 'wandb login' if not logged in",
            }
            
            logger.info(f"   Weights & Biases: ✅ {wandb.__version__}")
            logger.info(f"   W&B logged in: {'✅' if logged_in else '❌'}")
            
        except ImportError:
            checks["wandb_installed"] = {
                "passed": False,
                "note": "Required for experiment tracking",
            }
            logger.info("   Weights & Biases: ❌ not installed")
        
        # Check HuggingFace Hub
        try:
            from huggingface_hub import HfApi
            checks["huggingface_hub"] = {
                "passed": True,
                "note": "Required for model downloading",
            }
            logger.info("   HuggingFace Hub: ✅ available")
        except ImportError:
            checks["huggingface_hub"] = {
                "passed": False,
                "note": "Optional for model downloading",
            }
            logger.info("   HuggingFace Hub: ⚠️  not installed")
        
        # Check if can import project modules
        project_modules = [
            "pillar0.utils.plateau_ies",
            "pillar0.utils.efficiency_metrics",
            "pillar0.callbacks.ies_callback",
            "scripts.run_research_sweep",
        ]
        
        checks["project_modules"] = {}
        for module_name in project_modules:
            try:
                __import__(module_name.replace("/", "."))
                checks["project_modules"][module_name] = {
                    "passed": True,
                }
                logger.info(f"   Module {module_name}: ✅")
            except ImportError as e:
                checks["project_modules"][module_name] = {
                    "passed": False,
                    "error": str(e),
                }
                logger.info(f"   Module {module_name}: ❌")
        
        return checks
    
    def count_checks(self, d: Dict[str, Any]) -> Tuple[int, int]:
        """Recursively count passed and total checks."""
        passed = 0
        total = 0

        for v in d.values():
            if isinstance(v, dict):
                if "passed" in v:
                    total += 1
                    if v["passed"]:
                        passed += 1
                else:
                    p, t = self.count_checks(v)
                    passed += p
                    total += t

        return passed, total
    
    def generate_summary(self) -> Dict:
        """Generate validation summary."""
        summary = {
            "timestamp": self._get_timestamp(),
            "overall_status": self._calculate_overall_status(),
            "category_summary": {},
            "recommendations": [],
            "next_steps": [],
        }
        
        # Calculate status for each category using recursive counting
        for category, checks in self.results.items():
            passed, total = self.count_checks(checks)
            summary["category_summary"][category] = {
                "passed": passed,
                "total": total,
                "percentage": passed / total * 100 if total > 0 else 0,
            }
        
        # Generate recommendations
        summary["recommendations"] = self._generate_recommendations()
        
        # Generate next steps
        summary["next_steps"] = self._generate_next_steps(summary["overall_status"])
        
        return summary
    
    def _calculate_overall_status(self) -> bool:
        """Calculate overall validation status."""
        # Check critical categories
        critical_categories = ["basic_checks", "package_checks", "project_checks"]
        
        for category in critical_categories:
            if category in self.results:
                checks = self.results[category]
                # Recursively check if any critical check failed
                def check_failed(d, category_name=""):
                    for k, v in d.items():
                        if isinstance(v, dict):
                            if "passed" in v and not v["passed"]:
                                # Skip optional checks
                                optional_checks = ["flash_attention", "torch_compile", "huggingface_hub", "pytorch_sdpa"]
                                if k not in optional_checks and "optional" not in v:
                                    # For basic_checks category, disk_space might be optional
                                    if category_name == "basic_checks" and k == "disk_space":
                                        if v.get("optional", False):
                                            continue  # Skip optional disk space check
                                    return True
                            else:
                                if check_failed(v, category_name):
                                    return True
                    return False
                
                if check_failed(checks, category):
                    return False
        
        return True
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Package recommendations
        package_checks = self.results.get("package_checks", {})
        for package, check in package_checks.items():
            if not check.get("passed", False):
                required_version = check.get('required', '?')
                actual_version = check.get('actual', 'not installed')
                recommendations.append(
                    f"Install {package} >= {required_version} "
                    f"(currently: {actual_version})"
                )
        
        # Basic requirements recommendations
        basic_checks = self.results.get("basic_checks", {})
        disk_check = basic_checks.get("disk_space", {})
        if not disk_check.get("passed", False) and not disk_check.get("optional", False):
            recommendations.append(f"Free up disk space: {disk_check.get('actual', 'Check available storage')}")
        
        # GPU recommendations
        gpu_checks = self.results.get("gpu_checks", {})
        if not gpu_checks.get("cuda_available", {}).get("passed", False):
            recommendations.append(
                "Install CUDA and ensure GPU drivers are up to date"
            )
        
        # PyTorch SDPA recommendation
        sdpa_check = gpu_checks.get("pytorch_sdpa", {})
        if not sdpa_check.get("passed", False):
            recommendations.append("Upgrade to PyTorch 2.0+ for built-in optimized attention (SDPA)")
        
        # Project structure recommendations
        project_checks = self.results.get("project_checks", {})
        dir_checks = project_checks.get("directories", {})
        for dir_name, check in dir_checks.items():
            if not check.get("passed", False):
                recommendations.append(f"Create missing directory: {dir_name}")
        
        file_checks = project_checks.get("files", {})
        for file_name, check in file_checks.items():
            if not check.get("passed", False):
                recommendations.append(f"Create missing file: {file_name}")
        
        # Integration recommendations
        integration_checks = self.results.get("integration_checks", {})
        if not integration_checks.get("wandb_installed", {}).get("passed", False):
            recommendations.append("Install Weights & Biases: pip install wandb")
        
        if not integration_checks.get("wandb_logged_in", {}).get("passed", False):
            recommendations.append("Login to Weights & Biases: wandb login")
        
        return recommendations
    
    def _generate_next_steps(self, overall_status: bool) -> List[str]:
        """Generate next steps based on validation status."""
        if overall_status:
            gpu_checks = self.results.get("gpu_checks", {})
            sdpa_check = gpu_checks.get("pytorch_sdpa", {})
            
            steps = [
                "✅ Installation validated successfully!",
                "Next: Configure your experiments in configs/pillar0_research_2026.yaml",
                "Next: Run experiments: python scripts/run_research_sweep.py",
                "Next: Monitor results: wandb.ai",
            ]
            
            # Add note about PyTorch SDPA if available
            if sdpa_check.get("flash_available"):
                steps.insert(1, "✅ PyTorch Flash SDPA (built-in Flash Attention) is enabled!")
            elif sdpa_check.get("mem_efficient_available"):
                steps.insert(1, "✅ PyTorch Memory Efficient SDPA is enabled!")
            
            return steps
        else:
            return [
                "⚠️  Installation has issues that need to be fixed.",
                "Review the recommendations above and fix the issues.",
                "Run validation again after fixing issues: python scripts/validate_installation.py",
            ]
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def save_results(self, summary: Dict):
        """Save validation results to file."""
        # Ensure logs directory exists
        logs_dir = self.project_root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        results_path = logs_dir / "validation_results.json"
        
        full_results = {
            "summary": summary,
            "detailed_results": self.results,
        }
        
        with open(results_path, 'w') as f:
            json.dump(full_results, f, indent=2, default=str)
        
        # Save summary separately
        summary_path = logs_dir / "validation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"\n📁 Validation results saved:")
        logger.info(f"   Detailed: {results_path}")
        logger.info(f"   Summary: {summary_path}")
    
    def print_summary(self, summary: Dict):
        """Print validation summary."""
        logger.info(f"\n{'='*60}")
        logger.info("VALIDATION SUMMARY")
        logger.info(f"{'='*60}")
        
        # Overall status
        overall_passed = summary["overall_status"]
        status = "✅ PASSED" if overall_passed else "❌ FAILED"
        logger.info(f"\nOverall Status: {status}")
        
        # Category breakdown
        logger.info(f"\nCategory Breakdown:")
        for category, cat_summary in summary["category_summary"].items():
            percentage = cat_summary["percentage"]
            if percentage >= 90:
                status_symbol = "✅"
            elif percentage >= 70:
                status_symbol = "⚠️"
            else:
                status_symbol = "❌"
            logger.info(f"  {category}: {status_symbol} {percentage:.1f}% "
                       f"({cat_summary['passed']}/{cat_summary['total']})")
        
        # Recommendations - only show if validation failed
        if not overall_passed and summary["recommendations"]:
            logger.info(f"\n💡 Recommendations:")
            for rec in summary["recommendations"]:
                logger.info(f"  • {rec}")
        
        # Next steps
        logger.info(f"\n🚀 Next Steps:")
        for step in summary["next_steps"]:
            logger.info(f"  • {step}")
        
        logger.info(f"\n📁 Complete results saved in logs/validation_*.json")
        logger.info(f"{'='*60}")
        
        # Additional info if overall passed despite disk space warning
        if overall_passed and self.results.get("basic_checks", {}).get("disk_space", {}).get("warning", ""):
            logger.info(f"\n⚠️  Note: Disk space check reported issues but marked as optional.")
            logger.info(f"   Your installation is valid, but ensure you have sufficient storage for experiments.")
        
        # Highlight PyTorch SDPA status
        gpu_checks = self.results.get("gpu_checks", {})
        sdpa_check = gpu_checks.get("pytorch_sdpa", {})
        if sdpa_check.get("flash_available"):
            logger.info(f"\n✨ PyTorch Flash SDPA (built-in Flash Attention) is enabled!")
            logger.info(f"   Use torch.nn.functional.scaled_dot_product_attention() for optimized attention.")
        elif sdpa_check.get("mem_efficient_available"):
            logger.info(f"\n✨ PyTorch Memory Efficient SDPA is enabled!")
            logger.info(f"   Use torch.nn.functional.scaled_dot_product_attention() for optimized attention.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate installation")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick validation (skip GPU checks)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs/validation_results.json",
        help="Output file for validation results",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force validation to pass even with non-critical issues",
    )
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = InstallationValidator()
    
    # Run validation
    logger.info("Starting installation validation...")
    summary = validator.run_all_checks(args.quick)
    
    # Print summary
    validator.print_summary(summary)
    
    # Exit code based on validation status
    if not summary["overall_status"] and not args.force:
        logger.error("\n❌ Validation failed. Please fix the issues above.")
        logger.info("  Use --force to ignore non-critical issues and continue.")
        sys.exit(1)
    else:
        if args.force:
            logger.warning("\n⚠️  Validation completed with --force flag (ignoring some issues).")
        else:
            logger.info("\n🎉 Validation passed! Environment is ready for research.")
        sys.exit(0)


if __name__ == "__main__":
    main()