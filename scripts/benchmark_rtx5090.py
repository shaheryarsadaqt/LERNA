#!/usr/bin/env python3
"""
RTX 5090 Benchmarking Script

Benchmarks performance on RTX 5090 with various optimizations:
1. Flash Attention 2
2. BF16/FP16 mixed precision
3. Torch compile optimizations
4. Batch size scaling
"""

import torch
import numpy as np
import time
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import argparse
import logging

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


class RTX5090Benchmark:
    """Benchmark suite for RTX 5090."""
    
    def __init__(self, output_dir: str = "./experiments/benchmarks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # RTX 5090 specific features
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        self.cuda_version = torch.version.cuda if torch.cuda.is_available() else None
        
        logger.info(f"🔧 Initializing benchmark on: {self.gpu_name}")
        logger.info(f"   CUDA Version: {self.cuda_version}")
        logger.info(f"   PyTorch Version: {torch.__version__}")
        
        # Test models
        self.models = {
            "bert-base-uncased": "bert-base-uncased",
            "roberta-base": "roberta-base",
            "deberta-v3-base": "microsoft/deberta-v3-base",
            "modernbert-base": "answerdotai/ModernBERT-base",
        }
        
        # Benchmark configurations
        self.batch_sizes = [8, 16, 32, 64, 128]
        self.sequence_lengths = [64, 128, 256, 512]
        self.precision_modes = ["fp32", "fp16", "bf16"]
        
        # Results storage
        self.results = []
    
    def benchmark_model_loading(self) -> Dict:
        """Benchmark model loading time."""
        logger.info("\n📦 Benchmarking model loading...")
        
        results = {}
        for model_name, model_path in self.models.items():
            start_time = time.time()
            
            try:
                config = AutoConfig.from_pretrained(model_path)
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_path,
                    config=config,
                )
                model.to(self.device)
                model.eval()
                
                load_time = time.time() - start_time
                param_count = sum(p.numel() for p in model.parameters())
                
                results[model_name] = {
                    "load_time_seconds": load_time,
                    "parameters_millions": param_count / 1e6,
                    "memory_allocated_gb": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
                }
                
                logger.info(f"   {model_name}: {load_time:.2f}s, {param_count/1e6:.1f}M params")
                
                # Clean up
                del model
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.warning(f"   Failed to load {model_name}: {e}")
                results[model_name] = {"error": str(e)}
        
        return results
    
    def benchmark_inference(self, model_name: str, batch_size: int = 32, 
                          seq_length: int = 128, precision: str = "fp32") -> Dict:
        """Benchmark inference performance."""
        logger.info(f"\n🚀 Benchmarking inference: {model_name}, bs={batch_size}, seq={seq_length}, {precision}")
        
        try:
            # Load model
            model_path = self.models[model_name]
            config = AutoConfig.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                config=config,
            )
            
            # Set precision
            if precision == "fp16":
                model.half()
            elif precision == "bf16":
                model.bfloat16()
            
            model.to(self.device)
            model.eval()
            
            # Create dummy input
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length)).to(self.device)
            attention_mask = torch.ones_like(input_ids).to(self.device)
            
            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    _ = model(input_ids, attention_mask=attention_mask)
            
            torch.cuda.synchronize()
            
            # Benchmark
            times = []
            memory_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            with torch.no_grad():
                for _ in range(100):
                    start = time.time()
                    _ = model(input_ids, attention_mask=attention_mask)
                    torch.cuda.synchronize()
                    times.append(time.time() - start)
            
            memory_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            # Calculate statistics
            avg_time = np.mean(times)
            std_time = np.std(times)
            throughput = batch_size / avg_time
            
            result = {
                "model": model_name,
                "batch_size": batch_size,
                "sequence_length": seq_length,
                "precision": precision,
                "avg_inference_time_ms": avg_time * 1000,
                "std_inference_time_ms": std_time * 1000,
                "throughput_samples_per_second": throughput,
                "memory_usage_gb": (memory_after - memory_before) / 1e9,
                "parameters_millions": sum(p.numel() for p in model.parameters()) / 1e6,
            }
            
            logger.info(f"   Avg time: {avg_time*1000:.2f}ms, Throughput: {throughput:.1f} samples/s")
            
            # Clean up
            del model
            torch.cuda.empty_cache()
            
            return result
            
        except Exception as e:
            logger.error(f"   Inference benchmark failed: {e}")
            return {
                "model": model_name,
                "batch_size": batch_size,
                "sequence_length": seq_length,
                "precision": precision,
                "error": str(e),
            }
    
    def benchmark_training_step(self, model_name: str, batch_size: int = 32,
                              seq_length: int = 128, precision: str = "fp32") -> Dict:
        """Benchmark training step performance."""
        logger.info(f"\n🏋️  Benchmarking training step: {model_name}, bs={batch_size}, seq={seq_length}, {precision}")
        
        try:
            # Load model
            model_path = self.models[model_name]
            config = AutoConfig.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                config=config,
            )
            
            # Set precision
            if precision == "fp16":
                model.half()
            elif precision == "bf16":
                model.bfloat16()
            
            model.to(self.device)
            model.train()
            
            # Create dummy input
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length)).to(self.device)
            attention_mask = torch.ones_like(input_ids).to(self.device)
            labels = torch.randint(0, config.num_labels, (batch_size,)).to(self.device)
            
            # Setup optimizer
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
            
            # Warmup
            for _ in range(10):
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
            
            torch.cuda.synchronize()
            
            # Benchmark
            times = []
            memory_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            for _ in range(50):
                start = time.time()
                
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                torch.cuda.synchronize()
                times.append(time.time() - start)
            
            memory_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            # Calculate statistics
            avg_time = np.mean(times)
            std_time = np.std(times)
            throughput = batch_size / avg_time
            
            result = {
                "model": model_name,
                "batch_size": batch_size,
                "sequence_length": seq_length,
                "precision": precision,
                "avg_step_time_ms": avg_time * 1000,
                "std_step_time_ms": std_time * 1000,
                "throughput_samples_per_second": throughput,
                "memory_usage_gb": (memory_after - memory_before) / 1e9,
                "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
            }
            
            logger.info(f"   Avg step time: {avg_time*1000:.2f}ms, Throughput: {throughput:.1f} samples/s")
            
            # Clean up
            del model, optimizer
            torch.cuda.empty_cache()
            
            return result
            
        except Exception as e:
            logger.error(f"   Training benchmark failed: {e}")
            return {
                "model": model_name,
                "batch_size": batch_size,
                "sequence_length": seq_length,
                "precision": precision,
                "error": str(e),
            }
    
    def benchmark_flash_attention(self) -> Dict:
        """Benchmark Flash Attention 2 performance."""
        logger.info("\n⚡ Benchmarking Flash Attention 2...")
        
        try:
            # Check if flash attention is available
            try:
                from flash_attn import flash_attn_func
                flash_available = True
            except ImportError:
                flash_available = False
                logger.warning("   Flash Attention 2 not available. Install with: pip install flash-attn --no-build-isolation")
            
            if not flash_available:
                return {"available": False, "error": "Flash Attention 2 not installed"}
            
            # Test configuration
            batch_size = 32
            seq_length = 512
            num_heads = 12
            head_dim = 64
            
            # Create dummy tensors
            q = torch.randn(batch_size, seq_length, num_heads, head_dim, 
                          device=self.device, dtype=torch.float16)
            k = torch.randn(batch_size, seq_length, num_heads, head_dim,
                          device=self.device, dtype=torch.float16)
            v = torch.randn(batch_size, seq_length, num_heads, head_dim,
                          device=self.device, dtype=torch.float16)
            
            # Warmup
            for _ in range(10):
                _ = flash_attn_func(q, k, v)
            
            torch.cuda.synchronize()
            
            # Benchmark Flash Attention
            times_flash = []
            for _ in range(100):
                start = time.time()
                _ = flash_attn_func(q, k, v)
                torch.cuda.synchronize()
                times_flash.append(time.time() - start)
            
            # Benchmark standard attention for comparison
            times_standard = []
            for _ in range(100):
                start = time.time()
                
                # Standard attention implementation
                scale = 1.0 / (head_dim ** 0.5)
                attn = torch.matmul(q, k.transpose(-2, -1)) * scale
                attn = torch.softmax(attn, dim=-1)
                _ = torch.matmul(attn, v)
                
                torch.cuda.synchronize()
                times_standard.append(time.time() - start)
            
            # Calculate speedup
            avg_time_flash = np.mean(times_flash)
            avg_time_standard = np.mean(times_standard)
            speedup = avg_time_standard / avg_time_flash
            
            result = {
                "available": True,
                "batch_size": batch_size,
                "sequence_length": seq_length,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "avg_time_flash_ms": avg_time_flash * 1000,
                "avg_time_standard_ms": avg_time_standard * 1000,
                "speedup": speedup,
                "memory_savings_estimate": "2-4x",  # Flash Attention typically saves memory
            }
            
            logger.info(f"   Flash Attention: {avg_time_flash*1000:.2f}ms")
            logger.info(f"   Standard Attention: {avg_time_standard*1000:.2f}ms")
            logger.info(f"   Speedup: {speedup:.2f}x")
            
            return result
            
        except Exception as e:
            logger.error(f"   Flash Attention benchmark failed: {e}")
            return {"available": False, "error": str(e)}
    
    def benchmark_torch_compile(self) -> Dict:
        """Benchmark PyTorch 2.0 compile optimizations."""
        logger.info("\n🌀 Benchmarking Torch Compile...")
        
        try:
            # Test model
            model_path = self.models["bert-base-uncased"]
            config = AutoConfig.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                config=config,
            )
            model.to(self.device)
            model.eval()
            
            # Create dummy input
            batch_size = 32
            seq_length = 128
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length)).to(self.device)
            attention_mask = torch.ones_like(input_ids).to(self.device)
            
            # Benchmark without compile
            times_no_compile = []
            for _ in range(100):
                start = time.time()
                with torch.no_grad():
                    _ = model(input_ids, attention_mask=attention_mask)
                torch.cuda.synchronize()
                times_no_compile.append(time.time() - start)
            
            # Compile model
            compiled_model = torch.compile(model, mode="max-autotune")
            
            # Warmup for compiled model (compilation happens on first run)
            with torch.no_grad():
                _ = compiled_model(input_ids, attention_mask=attention_mask)
            
            torch.cuda.synchronize()
            
            # Benchmark with compile
            times_compile = []
            for _ in range(100):
                start = time.time()
                with torch.no_grad():
                    _ = compiled_model(input_ids, attention_mask=attention_mask)
                torch.cuda.synchronize()
                times_compile.append(time.time() - start)
            
            # Calculate speedup
            avg_time_no_compile = np.mean(times_no_compile)
            avg_time_compile = np.mean(times_compile)
            speedup = avg_time_no_compile / avg_time_compile
            
            result = {
                "model": "bert-base-uncased",
                "batch_size": batch_size,
                "sequence_length": seq_length,
                "avg_time_no_compile_ms": avg_time_no_compile * 1000,
                "avg_time_compile_ms": avg_time_compile * 1000,
                "speedup": speedup,
                "compile_mode": "max-autotune",
            }
            
            logger.info(f"   No compile: {avg_time_no_compile*1000:.2f}ms")
            logger.info(f"   With compile: {avg_time_compile*1000:.2f}ms")
            logger.info(f"   Speedup: {speedup:.2f}x")
            
            # Clean up
            del model, compiled_model
            torch.cuda.empty_cache()
            
            return result
            
        except Exception as e:
            logger.error(f"   Torch compile benchmark failed: {e}")
            return {"error": str(e)}
    
    def run_comprehensive_benchmark(self) -> Dict:
        """Run comprehensive benchmarking suite."""
        logger.info(f"\n{'='*60}")
        logger.info("STARTING COMPREHENSIVE RTX 5090 BENCHMARK")
        logger.info(f"{'='*60}")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "gpu_info": {
                "name": self.gpu_name,
                "cuda_version": self.cuda_version,
                "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0,
                "compute_capability": torch.cuda.get_device_capability() if torch.cuda.is_available() else None,
            },
            "benchmarks": {},
        }
        
        # 1. Model loading benchmark
        results["benchmarks"]["model_loading"] = self.benchmark_model_loading()
        
        # 2. Inference benchmarks
        inference_results = []
        for model_name in self.models.keys():
            for batch_size in [16, 32, 64]:
                for precision in ["fp32", "fp16", "bf16"]:
                    result = self.benchmark_inference(
                        model_name=model_name,
                        batch_size=batch_size,
                        seq_length=128,
                        precision=precision,
                    )
                    inference_results.append(result)
        
        results["benchmarks"]["inference"] = inference_results
        
        # 3. Training step benchmarks
        training_results = []
        for model_name in ["bert-base-uncased", "modernbert-base"]:
            for batch_size in [16, 32]:
                for precision in ["fp32", "fp16", "bf16"]:
                    result = self.benchmark_training_step(
                        model_name=model_name,
                        batch_size=batch_size,
                        seq_length=128,
                        precision=precision,
                    )
                    training_results.append(result)
        
        results["benchmarks"]["training"] = training_results
        
        # 4. Flash Attention benchmark
        results["benchmarks"]["flash_attention"] = self.benchmark_flash_attention()
        
        # 5. Torch compile benchmark
        results["benchmarks"]["torch_compile"] = self.benchmark_torch_compile()
        
        # 6. Memory efficiency analysis
        results["benchmarks"]["memory_efficiency"] = self.analyze_memory_efficiency()
        
        # Generate summary
        results["summary"] = self.generate_benchmark_summary(results)
        
        # Save results
        self.save_results(results)
        
        return results
    
    def analyze_memory_efficiency(self) -> Dict:
        """Analyze memory efficiency for different configurations."""
        logger.info("\n💾 Analyzing memory efficiency...")
        
        results = {}
        
        for model_name in self.models.keys():
            try:
                model_path = self.models[model_name]
                config = AutoConfig.from_pretrained(model_path)
                
                # Test different precisions
                memory_usage = {}
                
                for precision in ["fp32", "fp16", "bf16"]:
                    # Clear cache
                    torch.cuda.empty_cache()
                    
                    # Load model with specific precision
                    model = AutoModelForSequenceClassification.from_pretrained(
                        model_path,
                        config=config,
                    )
                    
                    if precision == "fp16":
                        model.half()
                    elif precision == "bf16":
                        model.bfloat16()
                    
                    model.to(self.device)
                    model.eval()
                    
                    # Create dummy input
                    input_ids = torch.randint(0, config.vocab_size, (32, 128)).to(self.device)
                    attention_mask = torch.ones_like(input_ids).to(self.device)
                    
                    # Measure memory
                    torch.cuda.reset_peak_memory_stats()
                    memory_before = torch.cuda.memory_allocated()
                    
                    with torch.no_grad():
                        _ = model(input_ids, attention_mask=attention_mask)
                    
                    memory_after = torch.cuda.memory_allocated()
                    peak_memory = torch.cuda.max_memory_allocated()
                    
                    memory_usage[precision] = {
                        "base_memory_gb": memory_before / 1e9,
                        "inference_memory_gb": memory_after / 1e9,
                        "peak_memory_gb": peak_memory / 1e9,
                        "activation_memory_gb": (memory_after - memory_before) / 1e9,
                    }
                    
                    # Clean up
                    del model
                    torch.cuda.empty_cache()
                
                results[model_name] = memory_usage
                
                logger.info(f"   {model_name}:")
                for precision, usage in memory_usage.items():
                    logger.info(f"     {precision}: peak={usage['peak_memory_gb']:.2f}GB, "
                              f"activation={usage['activation_memory_gb']:.2f}GB")
                
            except Exception as e:
                logger.error(f"   Memory analysis failed for {model_name}: {e}")
                results[model_name] = {"error": str(e)}
        
        return results
    
    def generate_benchmark_summary(self, results: Dict) -> Dict:
        """Generate benchmark summary with key findings."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "gpu": self.gpu_name,
            "key_findings": [],
            "recommendations": [],
            "performance_comparison": {},
        }
        
        # Extract key performance metrics
        inference_results = results["benchmarks"]["inference"]
        
        # Find best performing configuration
        if inference_results:
            # Group by model and find fastest
            model_performance = {}
            for result in inference_results:
                if "error" not in result:
                    model = result["model"]
                    throughput = result.get("throughput_samples_per_second", 0)
                    
                    if model not in model_performance or throughput > model_performance[model]["throughput"]:
                        model_performance[model] = {
                            "throughput": throughput,
                            "batch_size": result["batch_size"],
                            "precision": result["precision"],
                            "avg_time_ms": result.get("avg_inference_time_ms", 0),
                        }
            
            # Add to summary
            summary["performance_comparison"]["inference"] = model_performance
            
            # Identify fastest model
            fastest_model = max(model_performance.items(), 
                              key=lambda x: x[1]["throughput"], 
                              default=(None, {}))
            
            if fastest_model[0]:
                summary["key_findings"].append(
                    f"Fastest inference: {fastest_model[0]} with {fastest_model[1]['throughput']:.1f} samples/sec "
                    f"(batch={fastest_model[1]['batch_size']}, {fastest_model[1]['precision']})"
                )
        
        # Analyze training performance
        training_results = results["benchmarks"]["training"]
        if training_results:
            training_performance = {}
            for result in training_results:
                if "error" not in result:
                    model = result["model"]
                    throughput = result.get("throughput_samples_per_second", 0)
                    
                    if model not in training_performance or throughput > training_performance[model]:
                        training_performance[model] = throughput
            
            summary["performance_comparison"]["training"] = training_performance
        
        # Flash Attention results
        flash_results = results["benchmarks"]["flash_attention"]
        if flash_results.get("available", False):
            speedup = flash_results.get("speedup", 1)
            summary["key_findings"].append(
                f"Flash Attention 2 provides {speedup:.2f}x speedup for attention computation"
            )
            summary["recommendations"].append(
                "Enable Flash Attention 2 for models that support it"
            )
        
        # Torch compile results
        compile_results = results["benchmarks"]["torch_compile"]
        if "speedup" in compile_results:
            speedup = compile_results["speedup"]
            summary["key_findings"].append(
                f"Torch compile provides {speedup:.2f}x speedup for inference"
            )
            summary["recommendations"].append(
                "Use torch.compile() for production inference"
            )
        
        # Memory efficiency recommendations
        memory_results = results["benchmarks"]["memory_efficiency"]
        if memory_results:
            # Check memory savings from mixed precision
            for model_name, precisions in memory_results.items():
                if isinstance(precisions, dict) and "fp16" in precisions and "fp32" in precisions:
                    fp16_memory = precisions["fp16"]["peak_memory_gb"]
                    fp32_memory = precisions["fp32"]["peak_memory_gb"]
                    
                    if fp32_memory > 0:
                        memory_saving = (fp32_memory - fp16_memory) / fp32_memory * 100
                        if memory_saving > 30:
                            summary["key_findings"].append(
                                f"FP16 reduces memory by {memory_saving:.1f}% for {model_name}"
                            )
                            summary["recommendations"].append(
                                f"Use FP16 precision for {model_name} to save memory"
                            )
        
        # General recommendations
        summary["recommendations"].extend([
            "Use batch sizes that maximize GPU utilization (32-64 for RTX 5090)",
            "Enable BF16 for training when possible (better stability than FP16)",
            "Use gradient checkpointing for memory-intensive models",
            "Implement mixed precision training for 1.5-2x speedup",
        ])
        
        return summary
    
    def save_results(self, results: Dict):
        """Save benchmark results to file."""
        # Save detailed results
        detailed_path = self.output_dir / f"rtx5090_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(detailed_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary
        summary_path = self.output_dir / "benchmark_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(results["summary"], f, indent=2, default=str)
        
        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("BENCHMARK SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"GPU: {self.gpu_name}")
        
        for finding in results["summary"]["key_findings"]:
            logger.info(f"• {finding}")
        
        logger.info(f"\n💡 Recommendations:")
        for recommendation in results["summary"]["recommendations"]:
            logger.info(f"• {recommendation}")
        
        logger.info(f"\n📁 Results saved:")
        logger.info(f"   Detailed: {detailed_path}")
        logger.info(f"   Summary: {summary_path}")
        logger.info(f"{'='*60}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="RTX 5090 Benchmarking Suite")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./experiments/benchmarks",
        help="Output directory for benchmark results",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark (skips some tests)",
    )
    
    args = parser.parse_args()
    
    # Initialize benchmark
    benchmark = RTX5090Benchmark(output_dir=args.output_dir)
    
    # Run comprehensive benchmark
    results = benchmark.run_comprehensive_benchmark()
    
    logger.info("\n🎉 Benchmarking complete!")


if __name__ == "__main__":
    main()