# LERNA Phase 1.2 — DGX Research Context

**Platform:** Ubuntu 18.04.6 LTS (Bionic Beaver) on NVIDIA DGX Station  
**Kernel:** 4.15.0-161-generic #169-Ubuntu SMP  
**Driver:** NVIDIA 525.125.06  
**CUDA env:** `torch_cuda` (conda)  
**Offline bundle:** `~/ettin_hf_bundle`  

---

## System State: FROZEN

All APT operations, kernel upgrades, and official LERNA runs are suspended pending administrator-controlled GPU recovery.

### Critical finding

`nvidia-smi` fails on **GPU 08:00.0** with:

```
Unable to determine the device handle for GPU0000:08:00.0: Unknown Error
```

PCI enumeration shows:

```
07:00.0 VGA compatible controller [0300]: NVIDIA Corporation GV100GL [Tesla V100 DGXS 32GB] [10de:1db7] (rev a1)
07:00.1 Audio device [0403]: NVIDIA Corporation Device [10de:10f2] (rev a1)
08:00.0 VGA compatible controller [0300]: NVIDIA Corporation GV100GL [Tesla V100 DGXS 32GB] [10de:1db7] (rev ff)
08:00.1 Audio device [0403]: NVIDIA Corporation Device [10de:10f2] (rev ff)
0e:00.0 VGA compatible controller [0300]: NVIDIA Corporation GV100GL [Tesla V100 DGXS 32GB] [10de:1db7] (rev a1)
0e:00.1 Audio device [0403]: NVIDIA Corporation Device [10de:10f2] (rev a1)
0f:00.0 VGA compatible controller [0300]: NVIDIA Corporation GV100GL [Tesla V100 DGXS 32GB] [10de:1db7] (rev a1)
0f:00.1 Audio device [0403]: NVIDIA Corporation Device [10de:10f2] (rev a1)
```

- `rev ff` on `08:00.0` while other GPUs show `rev a1` indicates that device’s PCIe configuration space is returning all ones.
- This is consistent with a GPU that has **fallen off the bus**, lost link/power state, or is hardware-inaccessible.
- The loaded NVIDIA 525 driver and existing `/dev/nvidia*` nodes do **not** repair this condition.

### Immediate conclusion

This is an **acute GPU/driver/PCIe failure**, not a proven kernel incompatibility. Do not attempt kernel upgrades to resolve it.

---

## Administrator Sequence

All steps requiring `sudo` must be run on the DGX console with a password or configured NOPASSWD for the relevant commands.

### 1. Capture diagnostics before reboot

```bash
mkdir -p ~/dgx_gpu_failure_20260712
cd ~/dgx_gpu_failure_20260712

date | tee date.txt
uname -a | tee uname.txt
cat /etc/os-release | tee os-release.txt
cat /etc/dgx-release 2>/dev/null | tee dgx-release.txt
cat /proc/driver/nvidia/version 2>&1 | tee nvidia-driver-version.txt
nvidia-smi -q 2>&1 | tee nvidia-smi-q.txt
lspci -nn | grep -i -E 'nvidia|vga|3d' | tee nvidia-pci-devices.txt
lsmod | grep -E '^nvidia|nouveau' | tee loaded-gpu-modules.txt
dkms status 2>&1 | tee dkms-status.txt
ls -l /dev/nvidia* 2>&1 | tee nvidia-device-nodes.txt
ps -ef | grep -E 'python|cuda|nvidia' | grep -v grep | tee gpu-related-processes.txt

sudo nvidia-bug-report.sh
sudo dmesg -T > dmesg-before-reboot.txt
sudo journalctl -k -b > kernel-journal-before-reboot.txt
```

Preserve:

- `nvidia-bug-report.log.gz`
- `dmesg-before-reboot.txt`
- `kernel-journal-before-reboot.txt`
- all existing diagnostic text files in `~/dgx_gpu_failure_20260712/`

### 2. Controlled reboot

```bash
sudo reboot
```

### 3. Post-reboot verification (immediately after boot)

```bash
uname -r

lspci -nn | grep -i -E 'nvidia|vga|3d'
lspci -s 08:00.0 -nn

nvidia-smi -L
nvidia-smi --query-gpu=index,pci.bus_id,uuid,name,temperature.gpu,power.draw --format=csv,noheader

for gpu in 0 1 2 3; do
  echo "=== GPU $gpu ==="
  nvidia-smi -i "$gpu" --query-gpu=index,pci.bus_id,uuid,name,temperature.gpu,power.draw --format=csv,noheader
done
```

**Required outcome:** `08:00.0` must return a normal revision (not `rev ff`), and all four GPUs must appear in `nvidia-smi`.

### 4. CUDA enumeration test

```bash
source /home/sheheryar/miniconda3/etc/profile.d/conda.sh
conda activate torch_cuda

python3 - <<'PY'
import torch

print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index))
    tensor = torch.ones(1024, device=f"cuda:{index}")
    print("  tensor:", tensor.device, "sum:", tensor.sum().item())
PY
```

### 5. If reboot does not recover `08:00.0`

- **Do not** attempt `nvidia-smi --gpu-reset`.
- Have the administrator perform a **full shutdown** and physical cold power cycle.
- Leave the machine unpowered long enough for PCIe/GPU power rails to discharge.
- Start it and repeat the checks above.
- If the device remains inaccessible, open an NVIDIA hardware-support case with `nvidia-bug-report.log.gz`.

---

## Repository Correction

The administrator must remove or disable the Jammy source before any future APT operation.

```bash
sudo cp -a /etc/apt/sources.list /etc/apt/sources.list.before-repo-fix
```

Then comment out:

```
deb http://th.archive.ubuntu.com/ubuntu jammy main
```

Verify no other `jammy` or `focal` entries remain:

```bash
grep -RniE 'jammy|focal|bionic' /etc/apt/sources.list /etc/apt/sources.list.d
```

**Do not run `apt update` or install anything** until the repository list has been reviewed as a complete DGX configuration.

---

## Before Resuming LERNA

Even if reboot recovers the GPU:

1. Preserve the current official manifest at 800 pending runs.
2. Do not immediately launch the full matrix.
3. Repeat a GPU smoke run (e.g., `full_finetune` + `grad_norm` + `weight_freeze` on `sst2` with 512–1024 samples).
4. Monitor `dmesg` for new `NVRM` or `Xid` events.
5. Confirm the selected CUDA GPU and `nvidia-smi -i` telemetry refer to the same physical PCI bus ID.
6. Obtain the administrator’s kernel/OS decision.

The experiment code is ready; the physical GPU state and mixed operating-system repositories are now the blockers.

---

## Evidence Files

All evidence is on the DGX under `~/dgx_gpu_failure_20260712/`:

- `date.txt`
- `uname.txt`
- `os-release.txt`
- `dgx-release.txt`
- `nvidia-driver-version.txt`
- `nvidia-smi-q.txt`
- `nvidia-pci-devices.txt`
- `loaded-gpu-modules.txt`
- `dkms-status.txt`
- `nvidia-device-nodes.txt`
- `gpu-related-processes.txt`
- `apt-release-sources.txt`
- `kernel-package-policy.txt`
- `nvidia-bug-report.log.gz` (pending `sudo nvidia-bug-report.sh`)
- `dmesg-before-reboot.txt` (pending)
- `kernel-journal-before-reboot.txt` (pending)
