#!/usr/bin/env python3
"""
💀 HACKER MODE - ULTRA FAST VERSION 💀
All text in red with green progress bar at bottom
MINIMAL cooldown for maximum speed!
"""

import os
import sys
import time
import re
import subprocess
import threading
import queue
import argparse
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

def orchestrator_output(output_queue, cooldown):
    """Run orchestrator and put output lines into queue"""
    cmd = [
        sys.executable, "scripts/run_full_experiment.py",
        "--output-dir", "/ssd_xs/home/scvi383/scvi383/experiments/full_baseline",
        "--num-seeds", "10",
        "--cooldown", str(cooldown),
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
        # Put the line into queue for main thread to print
        output_queue.put(line)
    
    process.wait()
    output_queue.put(None)  # Signal end

def progress_monitor(stop_event):
    """Show green progress bar at bottom"""
    log_dir = "/ssd_xs/home/scvi383/scvi383/experiments/full_baseline"
    total_runs = 80
    completed = 0
    last_pos = 0
    start_time = time.time()
    
    # Wait for log file to exist
    while not os.path.exists(log_dir) and not stop_event.is_set():
        time.sleep(1)
    
    with tqdm(
        total=total_runs,
        desc=green("PROGRESS"),
        bar_format=f"{dark_red('{l_bar}')}{green('{bar}')}{dark_red('{r_bar}')}",
        colour='green',
        ncols=80,
        position=0,
        leave=True,
        file=sys.stderr  # Use stderr to avoid interfering with stdout
    ) as pbar:
        
        while not stop_event.is_set() and completed < total_runs:
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
                
                # Update ETA
                if completed > 0:
                    elapsed = time.time() - start_time
                    eta = (total_runs - completed) * (elapsed / completed)
                    eta_str = str(timedelta(seconds=int(eta)))
                    pbar.set_postfix_str(dark_red(f"ETA: {eta_str}"))
                
                time.sleep(1)
                
            except Exception:
                time.sleep(2)
    
    # Completion message
    if not stop_event.is_set():
        print(bold_red("\n" + "═" * 60), file=sys.stderr)
        print(bold_red("████████╗ █████╗ ███████╗██╗  ██╗    ██████╗  ██████╗ ███╗   ██╗███████╗██╗"), file=sys.stderr)
        print(bold_red("╚══██╔══╝██╔══██╗██╔════╝██║ ██╔╝    ██╔══██╗██╔═══██╗████╗  ██║██╔════╝██║"), file=sys.stderr)
        print(bold_red("   ██║   ███████║███████╗█████╔╝     ██║  ██║██║   ██║██╔██╗ ██║█████╗  ██║"), file=sys.stderr)
        print(bold_red("   ██║   ██╔══██║╚════██║██╔═██╗     ██║  ██║██║   ██║██║╚██╗██║██╔══╝  ╚═╝"), file=sys.stderr)
        print(bold_red("   ██║   ██║  ██║███████║██║  ██╗    ██████╔╝╚██████╔╝██║ ╚████║███████╗██╗"), file=sys.stderr)
        print(bold_red("   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝    ╚═════╝  ╚═════╝ ╚═╝  ╚═══╝╚══════╝╚═╝"), file=sys.stderr)
        print(bold_red("═" * 60), file=sys.stderr)

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Hacker mode monitor for LERNA experiments")
    parser.add_argument("--cooldown", type=int, default=2, help="Cooldown between runs in seconds (default: 2)")
    args = parser.parse_args()
    
    # Clear screen
    os.system('clear')
    
    # Print header
    print(HEADER)
    print(dark_red(f"  SYSTEM BOOT: {time.strftime('%Y-%m-%d %H:%M:%S')}"))
    print(dark_red("  TARGET: 80 EXPERIMENTS"))
    print(dark_red(f"  COOLDOWN: {args.cooldown}s ⚡⚡⚡"))
    print(dark_red("  STATUS: ACTIVE"))
    print(bold_red("═" * 60))
    print()  # Empty line for spacing
    
    # Create queue for output lines
    output_queue = queue.Queue()
    stop_event = threading.Event()
    
    # Start orchestrator thread with cooldown parameter
    orch_thread = threading.Thread(target=orchestrator_output, args=(output_queue, args.cooldown), daemon=True)
    orch_thread.start()
    
    # Start progress monitor in a separate thread
    monitor_thread = threading.Thread(target=progress_monitor, args=(stop_event,), daemon=True)
    monitor_thread.start()
    
    try:
        # Main thread: print output from orchestrator
        while True:
            try:
                line = output_queue.get(timeout=0.1)
                if line is None:  # End of output
                    break
                # Print all orchestrator output in red
                print(red(line), end='')
                sys.stdout.flush()
            except queue.Empty:
                continue
    except KeyboardInterrupt:
        print(bold_red("\n\n⚠️  SHUTDOWN REQUESTED - Finishing current run..."), file=sys.stderr)
        stop_event.set()
        time.sleep(2)
    
    # Wait for threads to finish
    orch_thread.join(timeout=5)
    print(bold_red("\n✅ SYSTEM SHUTDOWN COMPLETE"), file=sys.stderr)
