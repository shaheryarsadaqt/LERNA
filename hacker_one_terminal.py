#!/usr/bin/env python3
"""
💀 HACKER MODE - ALL TEXT IN RED WITH GREEN PROGRESS BAR 💀
Runs everything in ONE terminal with full red aesthetic
"""

import os
import sys
import time
import re
import subprocess
import threading
from datetime import timedelta
from tqdm import tqdm

# Colors
RED = '\033[91m'
BOLD_RED = '\033[1;91m'
DARK_RED = '\033[2;91m'
GREEN = '\033[92m'
RESET = '\033[0m'

def red(text):
    return f"{RED}{text}{RESET}"

def bold_red(text):
    return f"{BOLD_RED}{text}{RESET}"

def dark_red(text):
    return f"{DARK_RED}{text}{RESET}"

def green(text):
    return f"{GREEN}{text}{RESET}"

# Hacker ASCII header
HEADER = f"""
{bold_red('╔══════════════════════════════════════════════════════════╗')}
{bold_red('║  ██╗     ███████╗██████╗ ███╗   ██╗ █████╗             ║')}
{bold_red('║  ██║     ██╔════╝██╔══██╗████╗  ██║██╔══██╗            ║')}
{bold_red('║  ██║     █████╗  ██████╔╝██╔██╗ ██║███████║            ║')}
{bold_red('║  ██║     ██╔══╝  ██╔══██╗██║╚██╗██║██╔══██║            ║')}
{bold_red('║  ███████╗███████╗██║  ██║██║ ╚████║██║  ██║            ║')}
{bold_red('║  ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝            ║')}
{bold_red('╚══════════════════════════════════════════════════════════╝')}
"""

def orchestrator_output():
    """Run orchestrator and pipe all output through red filter"""
    cmd = [
        sys.executable, "scripts/run_full_experiment.py",
        "--output-dir", "/ssd_xs/home/scvi383/scvi383/experiments/full_baseline",
        "--num-seeds", "10",
        "--wandb"
    ]
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    for line in iter(process.stdout.readline, ''):
        # Make ALL output red
        sys.stdout.write(red(line))
        sys.stdout.flush()
    
    return process.wait()

def progress_monitor():
    """Show green progress bar at bottom"""
    log_dir = "/ssd_xs/home/scvi383/scvi383/experiments/full_baseline"
    total_runs = 80
    completed = 0
    last_pos = 0
    start_time = time.time()
    
    # Wait for log file to exist
    while not os.path.exists(log_dir):
        time.sleep(1)
    
    with tqdm(
        total=total_runs,
        desc=green("PROGRESS"),
        bar_format=f"{dark_red('{l_bar}')}{green('{bar}')}{dark_red('{r_bar}')}",
        colour='green',
        ncols=80,
        position=0,
        leave=True
    ) as pbar:
        
        while completed < total_runs:
            try:
                # Find latest log
                log_files = sorted([f for f in os.listdir(log_dir) if f.startswith("orchestrator_")])
                if not log_files:
                    time.sleep(2)
                    continue
                
                log_file = os.path.join(log_dir, log_files[-1])
                
                with open(log_file, 'r') as f:
                    f.seek(last_pos)
                    new_lines = f.readlines()
                    last_pos = f.tell()
                
                for line in new_lines:
                    if "Completed:" in line and "in" in line:
                        completed += 1
                        pbar.update(1)
                        
                        # Extract task for status
                        task = re.search(r'([a-z]+)_s\d+', line)
                        if task:
                            pbar.set_postfix_str(dark_red(f"LAST: {task.group(1)}"))
                    
                    elif "Starting:" in line:
                        run = re.search(r'Starting: (\S+)', line)
                        if run:
                            pbar.set_description(green(f"▶ {run.group(1)}"))
                    
                    elif "FAILED" in line:
                        pbar.set_postfix_str(dark_red("⚠ RETRYING"))
                
                # Update ETA in header occasionally
                if int(time.time()) % 5 == 0 and completed > 0:
                    elapsed = time.time() - start_time
                    eta = (total_runs - completed) * (elapsed / completed)
                    eta_str = str(timedelta(seconds=int(eta)))
                    pbar.set_postfix_str(dark_red(f"ETA: {eta_str}"))
                
                time.sleep(1)
                
            except Exception:
                time.sleep(2)
    
    # Completion message
    print(bold_red("\n" + "═" * 60))
    print(bold_red("████████╗ █████╗ ███████╗██╗  ██╗    ██████╗  ██████╗ ███╗   ██╗███████╗██╗"))
    print(bold_red("╚══██╔══╝██╔══██╗██╔════╝██║ ██╔╝    ██╔══██╗██╔═══██╗████╗  ██║██╔════╝██║"))
    print(bold_red("   ██║   ███████║███████╗█████╔╝     ██║  ██║██║   ██║██╔██╗ ██║█████╗  ██║"))
    print(bold_red("   ██║   ██╔══██║╚════██║██╔═██╗     ██║  ██║██║   ██║██║╚██╗██║██╔══╝  ╚═╝"))
    print(bold_red("   ██║   ██║  ██║███████║██║  ██╗    ██████╔╝╚██████╔╝██║ ╚████║███████╗██╗"))
    print(bold_red("   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝    ╚═════╝  ╚═════╝ ╚═╝  ╚═══╝╚══════╝╚═╝"))
    print(bold_red("═" * 60))

if __name__ == "__main__":
    # Clear screen
    os.system('clear')
    
    # Print header
    print(HEADER)
    print(dark_red(f"  SYSTEM BOOT: {time.strftime('%Y-%m-%d %H:%M:%S')}"))
    print(dark_red("  TARGET: 80 EXPERIMENTS"))
    print(dark_red("  STATUS: ACTIVE"))
    print(bold_red("═" * 60))
    
    # Start monitor in background
    monitor_thread = threading.Thread(target=progress_monitor, daemon=True)
    monitor_thread.start()
    
    # Run orchestrator (its output is already red)
    exit_code = orchestrator_output()
    sys.exit(exit_code)
