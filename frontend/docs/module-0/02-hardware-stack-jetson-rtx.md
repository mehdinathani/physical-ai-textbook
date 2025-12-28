---
sidebar_position: 2
---

# Hardware Stack: Jetson & RTX

The computational demands of Physical AI are extraordinary. A humanoid robot must process high-resolution sensor streams at real-time rates, run inference on multiple neural networks simultaneously, execute complex planning algorithms, and maintain precise motor control—all while operating on battery power with strict thermal constraints. This chapter explores the hardware platforms that make such computations possible, focusing on NVIDIA's Jetson family of edge computing devices and RTX GPUs for high-performance simulation.

## Understanding Computational Requirements

Before selecting hardware, we must understand what Physical AI actually requires from a compute perspective. Consider the perception pipeline alone: a humanoid robot might have cameras providing 1080p RGB at 30 Hz, depth sensors generating point clouds, inertial measurement units providing 6-DOF acceleration at 200 Hz, and force sensors on each limb. Processing this data requires:

**Computer Vision Processing**: Modern CNN-based vision models for object detection, semantic segmentation, and pose estimation typically require 10-100 TOPS (tera-operations per second) for real-time performance. More advanced transformer-based models can require significantly more.

**Depth Processing**: Processing depth maps and generating point clouds involves 3D convolutions and geometric computations. Real-time performance on VGA-resolution depth data typically requires 5-20 TOPS.

**Planning and Control**: Model predictive control (MPC) for bipedal locomotion requires solving optimization problems at 100-500 Hz. LLM-based planning adds additional inference requirements, typically 10-50 TOPS depending on model size and complexity.

**Sensor Fusion**: Combining data from multiple sensors with different rates and noise characteristics requires Kalman filters or particle filters, which are computationally tractable but must fit within control loops.

**Neural Network Inference**: Running multiple neural networks simultaneously—whether for perception, planning, or control—can consume 50-200 TOPS depending on model complexity and fusion strategies.

The key insight is that these computations must happen at the edge, on the robot itself. While cloud computing offers virtually unlimited resources, the latency of network communication makes cloud-based control unsuitable for anything beyond high-level planning. Physical AI requires edge computing with substantial compute density.

```python
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, List

@dataclass
class ComputeRequirement:
    """Represents compute requirements for a subsystem."""
    name: str
    tops_required: float  # TOPS for neural network inference
    memory_gb: float      # GPU memory requirement
    power_watts: float    # Typical power consumption
    latency_budget_ms: float  # Maximum latency for real-time operation

class ComputeBudget:
    """
    Manages compute budget allocation across robot subsystems.
    Ensures total requirements don't exceed hardware capabilities.
    """

    def __init__(self, total_tops: float, total_memory_gb: float,
                 total_power_budget: float):
        self.total_tops = total_tops
        self.total_memory = total_memory_gb
        self.total_power = total_power_budget

        self.allocated: List[ComputeRequirement] = []
        self.remaining_tops = total_tops
        self.remaining_memory = total_memory_gb
        self.remaining_power = total_power_budget

    def allocate(self, req: ComputeRequirement) -> bool:
        """
        Attempt to allocate resources for a subsystem.
        Returns True if allocation successful, False otherwise.
        """
        if (req.tops_required > self.remaining_tops or
            req.memory_gb > self.remaining_memory or
            req.power_watts > self.remaining_power):
            return False

        self.allocated.append(req)
        self.remaining_tops -= req.tops_required
        self.remaining_memory -= req.memory_gb
        self.remaining_power -= req.power_watts

        return True

    def summary(self) -> Dict:
        """Return summary of resource allocation."""
        return {
            'total_tops': self.total_tops,
            'allocated_tops': self.total_tops - self.remaining_tops,
            'remaining_tops': self.remaining_tops,
            'total_memory_gb': self.total_memory,
            'allocated_memory': self.total_memory - self.remaining_memory,
            'total_power_watts': self.total_power,
            'allocated_power': self.total_power - self.remaining_power,
            'subsystems': [r.name for r in self.allocated]
        }

    def can_run_parallel(self, requirements: List[ComputeRequirement]) -> bool:
        """Check if a set of subsystems can run simultaneously."""
        total_tops = sum(r.tops_required for r in requirements)
        total_memory = sum(r.memory_gb for r in requirements)
        total_power = sum(r.power_watts for r in requirements)

        return (total_tops <= self.total_tops and
                total_memory <= self.total_memory and
                total_power <= self.total_power)

# Example compute budget for Jetson AGX Orin (200 TOPS mode)
budget = ComputeBudget(total_tops=200, total_memory_gb=64, total_power_budget=60)

# Typical humanoid robot requirements
perception_req = ComputeRequirement(
    name="Perception",
    tops_required=80,
    memory_gb=8,
    power_watts=25,
    latency_budget_ms=33  # 30 Hz
)

planning_req = ComputeRequirement(
    name="Planning",
    tops_required=40,
    memory_gb=4,
    power_watts=15,
    latency_budget_ms=100  # 10 Hz
)

control_req = ComputeRequirement(
    name="Control",
    tops_required=10,
    memory_gb=2,
    power_watts=8,
    latency_budget_ms=10  # 100 Hz
)

vocals_req = ComputeRequirement(
    name="Voice/ASR",
    tops_required=5,
    memory_gb=1,
    power_watts=3,
    latency_budget_ms=150
)

print("Allocating subsystems...")
budget.allocate(perception_req)
budget.allocate(planning_req)
budget.allocate(control_req)
budget.allocate(vocals_req)

print(budget.summary())
```

## NVIDIA Jetson Platform Overview

NVIDIA's Jetson platform provides a range of edge AI computing solutions optimized for power efficiency and thermal envelope. The platform shares software infrastructure across all devices, allowing you to develop on one Jetson and deploy to others as your requirements evolve.

### The Jetson Family Tree

The Jetson lineup spans from ultra-low-power modules for battery-powered devices to workstation-class systems for complex perception pipelines:

**Jetson Nano** represents the entry point, offering 472 GFLOPS (FP16) at under 10 watts. While insufficient for modern transformer-based perception, the Nano remains useful for simpler robotics applications, educational projects, and early-stage prototyping. Its 128-core Maxwell GPU and quad-core ARM CPU provide enough compute for basic CV tasks and simple navigation.

**Jetson TX2** bridges the gap with 1.33 TFLOPS (FP16) at under 15 watts. The TX2's Pascal-architecture GPU and Denver CPU cores enable real-time object detection and basic semantic segmentation. Many production robots deployed in the 2018-2021 timeframe used TX2 as their compute platform.

**Jetson Xavier NX** brings Volta-architecture GPU cores to a compact module, delivering 21 TOPS at 10-20 watts. The Xavier NX's tensor cores accelerate INT8 and FP16 inference, making it suitable for modern CNN-based perception. Its 6-core Carmel CPU handles higher-level planning tasks effectively.

**Jetson AGX Orin** represents the current flagship for edge computing, delivering up to 275 TOPS at 60 watts. The Orin architecture includes next-generation tensor cores, improved GPU cores based on Ampere architecture, and a dedicated Deep Learning Accelerator (DLA). The AGX Orin can run multiple complex perception models simultaneously while still meeting power and thermal constraints for mobile robots.

**Jetson Orin Nano** brings much of the Orin architecture to a lower price and power point, offering up to 40 TOPS at 7-25 watts. For many humanoid applications, the Orin Nano provides the best balance of performance and efficiency.

```python
import numpy as np
from dataclasses import dataclass

@dataclass
class JetsonSpec:
    """Specifications for a Jetson module."""
    name: str
    tops: float
    gpu_cores: int
    cpu_cores: int
    memory_gb: int
    power_watts: float
    architecture: str

# Jetson family specifications
JETSON_FAMILY = {
    'nano': JetsonSpec(
        name="Jetson Nano",
        tops=0.5,
        gpu_cores=128,
        cpu_cores=4,
        memory_gb=4,
        power_watts=10,
        architecture="Maxwell"
    ),
    'tx2': JetsonSpec(
        name="Jetson TX2",
        tops=1.33,
        gpu_cores=256,
        cpu_cores=6,
        memory_gb=8,
        power_watts=15,
        architecture="Pascal"
    ),
    'xavier_nx': JetsonSpec(
        name="Jetson Xavier NX",
        tops=21,
        gpu_cores=384,
        cpu_cores=6,
        memory_gb=8,
        power_watts=20,
        architecture="Volta"
    ),
    'orin_nano': JetsonSpec(
        name="Jetson Orin Nano",
        tops=40,
        gpu_cores=1024,
        cpu_cores=6,
        memory_gb=8,
        power_watts=25,
        architecture="Ampere"
    ),
    'agx_orin': JetsonSpec(
        name="Jetson AGX Orin",
        tops=275,
        gpu_cores=2048,
        cpu_cores=12,
        memory_gb=64,
        power_watts=60,
        architecture="Ampere"
    )
}

class JetsonSelector:
    """
    Selects appropriate Jetson module based on requirements.
    """

    def __init__(self, requirements: dict):
        self.min_tops = requirements.get('min_tops', 10)
        self.min_memory = requirements.get('min_memory_gb', 8)
        self.max_power = requirements.get('max_power_watts', 30)
        self.min_gpu_cores = requirements.get('min_gpu_cores', 256)

    def select(self) -> str:
        """Select the best Jetson module for requirements."""
        candidates = []

        for key, spec in JETSON_FAMILY.items():
            if (spec.tops >= self.min_tops and
                spec.memory_gb >= self.min_memory and
                spec.power_watts <= self.max_power and
                spec.gpu_cores >= self.min_gpu_cores):
                candidates.append((spec.tops / spec.power_watts, spec.name, key))

        if not candidates:
            return "No suitable Jetson module found"

        # Select highest performance per watt
        candidates.sort(reverse=True)
        return candidates[0][1]

    def compare(self) -> str:
        """Generate comparison of all modules."""
        comparison = []
        for key, spec in JETSON_FAMILY.items():
            efficiency = spec.tops / spec.power_watts if spec.power_watts > 0 else 0
            comparison.append(
                f"{spec.name:20} | "
                f"{spec.tops:6.1f} TOPS | "
                f"{spec.gpu_cores:4} cores | "
                f"{spec.power_watts:5} W | "
                f"{efficiency:5.1f} TOPS/W"
            )
        return "\n".join(comparison)

# Example usage
requirements = {
    'min_tops': 100,
    'min_memory_gb': 16,
    'max_power_watts': 60,
    'min_gpu_cores': 1024
}

selector = JetsonSelector(requirements)
print("Available Jetson modules:")
print(selector.compare())
print(f"\nSelected for requirements: {selector.select()}")
```

### Jetson Software Stack

All Jetson modules share a common software foundation built on Ubuntu Linux with NVIDIA's JetPack SDK. The key components include:

**CUDA Toolkit** provides GPU programming capability. Modern robotics code often uses CUDA directly or through libraries like CuDNN (deep neural network acceleration) and CuFFT (fast Fourier transforms for signal processing).

**TensorRT** is NVIDIA's inference optimizer and runtime. TensorRT takes trained neural networks and optimizes them for deployment, applying techniques like kernel auto-tuning, layer fusion, and precision calibration. A model optimized with TensorRT can run 2-10x faster than the raw framework implementation.

**VPI (Vision Programming Interface)** provides optimized implementations of computer vision algorithms optimized for Jetson's hardware accelerators, including the PVA (Programmable Vision Accelerator) and GPU.

**CUDA-X AI** libraries include optimized implementations of common AI operations—from basic linear algebra through specialized robotics algorithms like optical flow and 3D point cloud processing.

```python
import torch
import tensorrt as trt
import numpy as np
from pathlib import Path
from typing import Optional

class TensorRTConverter:
    """
    Converts PyTorch models to TensorRT engines for Jetson deployment.
    """

    def __init__(self, precision: str = 'fp16', workspace_mb: int = 4096):
        self.precision = precision
        self.workspace_mb = workspace_mb
        self.logger = trt.Logger(trt.Logger.WARNING)

    def convert(self, model: torch.nn.Module, input_shape: tuple,
                output_path: str) -> str:
        """
        Convert a PyTorch model to TensorRT engine.
        """
        # Export to ONNX first
        onnx_path = output_path.replace('.engine', '.onnx')
        self._export_onnx(model, input_shape, onnx_path)

        # Build TensorRT engine
        builder = trt.Builder(self.logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        parser = trt.OnnxParser(network, self.logger)

        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print(f"ONNX Parser Error {error}: {parser.get_error(error)}")
                raise RuntimeError("Failed to parse ONNX model")

        # Configure builder
        config = builder.create_builder_config()
        config.max_workspace_size = self.workspace_mb * 1024 * 1024

        if self.precision == 'fp16' and builder.platform_has_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        # Build engine
        engine = builder.build_engine(network, config)

        # Save engine
        with open(output_path, 'wb') as f:
            f.write(engine.serialize())

        return output_path

    def _export_onnx(self, model: torch.nn.Module, input_shape: tuple,
                     output_path: str):
        """Export model to ONNX format."""
        model.eval()
        dummy_input = torch.randn(input_shape)

        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            opset_version=13
        )

class ModelOptimizer:
    """
    Optimizes neural network models for Jetson deployment.
    Implements techniques for reducing model size and improving inference speed.
    """

    def __init__(self, target_device: str = 'agx_orin'):
        self.target = target_device

    def apply_quantization(self, model: torch.nn.Module,
                           calibration_data: np.ndarray) -> torch.nn.Module:
        """
        Apply dynamic quantization to reduce model size.
        """
        # Dynamic quantization for linear layers
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear, torch.nn.LSTM},
            dtype=torch.qint8
        )
        return quantized_model

    def prune_model(self, model: torch.nn.Module,
                    sparsity: float = 0.3) -> torch.nn.Module:
        """
        Apply magnitude-based pruning to reduce model parameters.
        """
        import torch.nn.utils.prune as prune

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=sparsity)

        return model

    def benchmark_inference(self, model: torch.nn.Module,
                            input_tensor: torch.Tensor,
                            num_warmup: int = 10,
                            num_runs: int = 100) -> dict:
        """
        Benchmark model inference time.
        """
        model.eval()
        device = next(model.parameters()).device

        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(input_tensor.to(device))

        # Benchmark
        import time
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = model(input_tensor.to(device))
                times.append(time.perf_counter() - start)

        return {
            'mean_ms': np.mean(times) * 1000,
            'std_ms': np.std(times) * 1000,
            'p95_ms': np.percentile(times, 95) * 1000,
            'throughput_fps': 1.0 / np.mean(times)
        }
```

### JetPack and L4T

**JetPack** is NVIDIA's comprehensive SDK for Jetson, bundling the operating system, drivers, libraries, and development tools. When installing a new Jetson module, you flash it with a JetPack image that includes Ubuntu 20.04 or 22.04, CUDA, TensorRT, cuDNN, VPI, and associated libraries.

**L4T (Linux for Tegra)** is the underlying Linux distribution that JetPack installs. L4T includes the kernel, device trees, and firmware specific to Jetson's Tegra processor. Understanding L4T is important when debugging boot issues, customizing the kernel, or working with custom carrier boards.

For development, you typically want the latest stable JetPack version that supports your module. As of this writing, JetPack 6.0 supports Jetson Orin modules with Ubuntu 22.04 and CUDA 12.2. JetPack 5.x supports older modules like Xavier NX and TX2 with Ubuntu 18.04 or 20.04.

## NVIDIA RTX Platform for Simulation

While Jetson handles on-robot computation, simulation workloads run on much more powerful hardware. NVIDIA RTX GPUs provide the parallel compute throughput necessary for physics simulation, rendering, and training neural networks.

### RTX Architecture for Simulation

RTX GPUs are based on NVIDIA's Ampere, Ada Lovelace, or Hopper architectures, each providing substantial improvements over previous generations for simulation workloads:

**CUDA Cores** provide general-purpose parallel compute. An RTX 4090 features over 16,000 CUDA cores, enabling massive parallelization of physics simulations and neural network training.

**RT Cores** accelerate ray tracing, which is essential for photorealistic rendering in simulation. When simulating cameras, RT cores enable physically accurate lighting, shadows, and reflections—critical for developing and testing vision algorithms.

**Tensor Cores** accelerate matrix operations for neural network inference and training. For sim-to-real transfer, tensor cores speed up both policy training and online adaptation.

**Large Frame Buffers** on RTX cards (up to 24GB on RTX 4090) enable higher-resolution simulation and more complex scenes without memory bottlenecks.

```python
import subprocess
import re

class RTXDiagnostics:
    """
    Diagnostic utilities for RTX GPU systems.
    """

    @staticmethod
    def get_gpu_info() -> dict:
        """Query GPU information using nvidia-smi."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True
            )
            lines = result.stdout.strip().split('\n')
            gpus = []
            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 3:
                    gpus.append({
                        'name': parts[0],
                        'memory_mb': int(parts[1]),
                        'driver': parts[2]
                    })
            return {'gpus': gpus, 'count': len(gpus)}
        except Exception as e:
            return {'error': str(e)}

    @staticmethod
    def get_utilization() -> dict:
        """Query current GPU utilization."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,temperature.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True
            )
            lines = result.stdout.strip().split('\n')
            utilizations = []
            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 3:
                    utilizations.append({
                        'utilization_percent': int(parts[0]),
                        'memory_used_mb': int(parts[1]),
                        'temperature_c': int(parts[2])
                    })
            return {'gpus': utilizations}
        except Exception as e:
            return {'error': str(e)}

    @staticmethod
    def check_cuda_version() -> dict:
        """Check CUDA version and capabilities."""
        try:
            # Check nvcc version
            result = subprocess.run(
                ['nvcc', '--version'],
                capture_output=True,
                text=True
            )
            version_match = re.search(r'release (\d+\.\d+)', result.stdout)
            cuda_version = version_match.group(1) if version_match else None

            # Check cuDNN version
            result = subprocess.run(
                ['dpkg', '-l', 'libcudnn8'],
                capture_output=True,
                text=True
            )
            cudnn_match = re.search(r'(\d+\.\d+\.\d+)', result.stdout)

            return {
                'cuda_version': cuda_version,
                'cudnn_version': cudnn_match.group(1) if cudnn_match else None
            }
        except Exception as e:
            return {'error': str(e)}

# Example usage
if __name__ == "__main__":
    diagnostics = RTXDiagnostics()

    print("GPU Information:")
    info = diagnostics.get_gpu_info()
    for gpu in info.get('gpus', []):
        print(f"  - {gpu['name']}: {gpu['memory_mb']}MB")

    print("\nCurrent Utilization:")
    util = diagnostics.get_utilization()
    for i, gpu in enumerate(util.get('gpus', [])):
        print(f"  GPU {i}: {gpu['utilization_percent']}% util, "
              f"{gpu['memory_used_mb']}MB used, {gpu['temperature_c']}C")
```

### Choosing Simulation Hardware

For serious simulation work, you need substantial GPU resources. Here are common configurations:

**RTX 3060/4060 (12GB)**: Minimum viable for basic simulation. Can handle simple scenes with a few robots at reduced resolution. Suitable for learning and early development.

**RTX 4080/4090 (16-24GB)**: Recommended for serious development. Handles complex scenes with multiple robots, high-resolution cameras, and physics simulation. The 24GB frame buffer on RTX 4090 is particularly valuable for large neural network training.

**RTX A6000 (48GB)**: Professional-grade option for research labs. The large frame buffer enables training on batch sizes that would otherwise require distributed training.

**Multi-GPU Workstations**: For complex projects, multiple GPUs enable parallel simulation (running multiple environments simultaneously) and larger neural network training.

### Isaac Sim Requirements

NVIDIA Isaac Sim has specific hardware requirements. The minimum configuration includes:

- GPU: NVIDIA RTX 3060 or better with 8GB VRAM
- CPU: Intel Core i7 or AMD Ryzen 7
- RAM: 32GB system memory
- Storage: 50GB SSD

For optimal performance, Isaac Sim recommends:

- GPU: RTX 4080/4090 with 16GB+ VRAM
- CPU: Intel Core i9 or AMD Ryzen 9
- RAM: 64GB+ system memory
- Storage: NVMe SSD with 100GB+ free space

Isaac Sim leverages RTX features heavily. RT cores enable physically accurate camera simulation, tensor cores accelerate ML workloads, and CUDA cores handle general physics and rendering.

## Hardware Selection for Humanoid Robots

Selecting hardware for a humanoid robot involves balancing competing constraints: compute requirements, power consumption, thermal management, cost, and physical size.

### Power Consumption Analysis

Power is often the limiting factor for mobile robots. A humanoid robot's battery capacity typically ranges from 500Wh to 2kWh, and the compute system must operate within this budget while leaving power for actuators.

Consider a typical humanoid with 20 degrees of freedom. High-performance servomotors might consume 50-100W each during aggressive motion, meaning the actuation system can draw 1-2kW during peak activity. With a 1kWh battery, you have perhaps 30-60 minutes of operation if the compute system is efficient.

Jetson power consumption scales with performance:

| Module | Peak Power | Typical Power | TOPS |
|--------|------------|---------------|------|
| Jetson Nano | 10W | 5W | 0.5 |
| Jetson TX2 | 15W | 10W | 1.3 |
| Jetson Xavier NX | 20W | 10W | 21 |
| Jetson Orin Nano | 25W | 7W | 40 |
| Jetson AGX Orin | 60W | 30W | 200-275 |

For a humanoid robot, Jetson AGX Orin or Jetson Orin NX are typically the right choices. Orin Nano is viable for simpler behaviors or as a secondary processor for specific tasks.

```python
import numpy as np
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class PowerBudget:
    """Represents power budget for robot subsystems."""
    battery_wh: float          # Battery capacity in Watt-hours
    compute_watts: float       # Compute system power
    actuation_watts: float     # Actuation system power
    overhead_watts: float      # Other systems (sensors, communications)

    def remaining_for_compute(self) -> float:
        """Compute remaining power budget for compute system."""
        total_other = self.actuation_watts + self.overhead_watts
        return self.battery_wh * 0.8 - total_other  # 80% of battery, reserve 20%

    def compute_runtime(self, compute_power: float) -> float:
        """Estimate runtime at given compute power."""
        available = self.battery_wh - self.actuation_watts - self.overhead_watts
        return available / compute_power

@dataclass
class ThermalProfile:
    """Thermal characteristics of compute system."""
    passive_cooling_capable: bool
    max_sustained_power_watts: float
    peak_power_watts: float
    thermal_throttling_threshold_c: float

class PowerManager:
    """
    Manages power allocation across robot subsystems.
    Implements dynamic power management for optimal performance.
    """

    def __init__(self, battery_wh: float, max_actuation_watts: float):
        self.battery_wh = battery_wh
        self.max_actuation = max_actuation_watts
        self.current_compute_power = 30  # Default for AGX Orin
        self.thermal_profile = ThermalProfile(
            passive_cooling_capable=False,
            max_sustained_power_watts=45,
            peak_power_watts=60,
            thermal_throttling_threshold_c=85
        )
        self.current_temperature = 45  # Initial temperature in Celsius
        self.safety_margin = 0.15  # 15% safety margin

    def set_performance_mode(self, mode: str) -> None:
        """
        Set Jetson power mode (affects performance/power trade-off).
        """
        modes = {
            'maxn': 60,    # Maximum performance
            'high': 45,    # High performance
            'medium': 30,  # Medium performance
            'low': 15,     # Low power
            'minimum': 7   # Minimum power
        }
        self.current_compute_power = modes.get(mode, 30)

    def get_actual_power_budget(self) -> Dict:
        """
        Get actual available power considering thermal constraints.
        """
        # Thermal throttling
        if self.current_temperature > self.thermal_profile.thermal_throttling_threshold_c:
            # Reduce power due to thermal throttling
            throttling_factor = max(0.5, 1.0 -
                (self.current_temperature - self.thermal_profile.thermal_throttling_threshold_c) / 20)
            effective_power = self.current_compute_power * throttling_factor
        else:
            effective_power = self.current_compute_power

        # Apply safety margin
        safe_power = effective_power * (1 - self.safety_margin)

        return {
            'requested_power_watts': self.current_compute_power,
            'effective_power_watts': effective_power,
            'safe_power_watts': safe_power,
            'thermal_status': 'throttled' if effective_power < self.current_compute_power else 'normal'
        }

    def estimate_runtime(self, average_actuation_power: float) -> float:
        """
        Estimate robot runtime with current configuration.
        """
        budget = PowerBudget(
            battery_wh=self.battery_wh,
            compute_watts=self.current_compute_power,
            actuation_watts=average_actuation_power,
            overhead_watts=10  # Sensors, communications
        )
        return budget.compute_runtime(self.current_compute_power)

# Example power analysis for humanoid robot
power_manager = PowerManager(battery_wh=1000, max_actuation_watts=1500)

print("Power Analysis for Humanoid Robot")
print("=" * 40)
print(f"Battery: {power_manager.battery_wh} Wh")
print(f"Max actuation: {power_manager.max_actuation} W")
print()

# Test different performance modes
for mode in ['maxn', 'high', 'medium', 'low']:
    power_manager.set_performance_mode(mode)
    status = power_manager.get_actual_power_budget()
    runtime = power_manager.estimate_runtime(average_actuation_power=500)
    print(f"Mode: {mode}")
    print(f"  Compute power: {status['effective_power_watts']:.0f} W")
    print(f"  Runtime (500W actuation): {runtime:.1f} hours")
```

### Thermal Considerations

Thermal management is critical for reliable operation. Jetson modules generate substantial heat under sustained load, and enclosed robot bodies make heat dissipation challenging.

**Passive Cooling**: Small robots or those operating in controlled environments can use passive cooling with heatsinks. This is silent and reliable but limits sustained performance.

**Active Cooling**: Fans provide higher cooling capacity but consume power, generate noise, and introduce moving parts that can fail. Many production robots use low-profile fans with controlled speed curves.

**Liquid Cooling**: For high-power applications, liquid cooling provides the most effective thermal management. However, it adds complexity, potential leak points, and weight.

When designing a robot's thermal system, you must consider not just peak heat generation but sustained operation. A robot might handle brief compute-intensive tasks, but hours of continuous navigation and perception create thermal challenges.

```python
from dataclasses import dataclass
from typing import Callable

@dataclass
class ThermalZone:
    """A thermal zone with temperature sensor and cooling control."""
    name: str
    current_temp_c: float
    target_temp_c: float
    max_temp_c: float
    cooling_device: str  # 'fan', 'pump', 'heatsink'

class ThermalManager:
    """
    Manages thermal state across robot subsystems.
    Implements thermal-aware scheduling and cooling control.
    """

    def __init__(self, zones: list = None):
        self.zones = zones or []
        self.thermal_time_constant = 30  # seconds to reach equilibrium
        self.ambient_temp = 25  # Celsius

    def add_zone(self, zone: ThermalZone):
        """Add a thermal zone to manage."""
        self.zones.append(zone)

    def update_temperature(self, zone_name: str, ambient_change: float = 0,
                          workload_factor: float = 1.0) -> float:
        """
        Update temperature for a zone based on workload and cooling.
        """
        zone = next((z for z in self.zones if z.name == zone_name), None)
        if zone is None:
            return 0

        # Heat generation proportional to workload
        heat_generation = 0.5 * workload_factor  # degrees per second at full load

        # Cooling proportional to temperature difference
        cooling_rate = (zone.current_temp_c - self.ambient_temp) / self.thermal_time_constant

        # Apply cooling device effect
        if zone.cooling_device == 'fan':
            cooling_rate *= 2.0
        elif zone.cooling_device == 'pump':
            cooling_rate *= 3.0

        # Update temperature
        delta_t = (heat_generation - cooling_rate + ambient_change / self.thermal_time_constant)
        zone.current_temp_c = max(zone.current_temp_c + delta_t, self.ambient_temp)

        # Safety check
        if zone.current_temp_c > zone.max_temp_c:
            self._trigger_thermal_emergency(zone)

        return zone.current_temp_c

    def get_cooling_level(self, zone_name: str) -> float:
        """Get current cooling level (0-1) for a zone."""
        zone = next((z for z in self.zones if z.name == zone_name), None)
        if zone is None:
            return 0

        temp_below_target = (zone.max_temp_c - zone.current_temp_c) / zone.max_temp_c
        return min(1.0, temp_below_target * 2)

    def thermal_aware_schedule(self, tasks: list,
                               current_temps: dict) -> list:
        """
        Schedule tasks considering thermal state.
        Returns ordered list of tasks with thermal delays if needed.
        """
        scheduled = []
        for task in tasks:
            task_temp = current_temps.get(task['zone'], 45)  # Default to moderate temp
            zone = next((z for z in self.zones if z.name == task['zone']), None)

            if zone and task_temp > zone.target_temp_c:
                # Delay task until temperature cools
                cool_time = (task_temp - zone.target_temp_c) * self.thermal_time_constant / 2
                task['delay_seconds'] = cool_time
            else:
                task['delay_seconds'] = 0

            scheduled.append(task)

        return scheduled

    def _trigger_thermal_emergency(self, zone: ThermalZone):
        """Handle thermal emergency - critical temperature exceeded."""
        print(f"THERMAL EMERGENCY: {zone.name} at {zone.current_temp_c:.1f}C")
        print("Actions: Throttling compute, enabling maximum cooling")
        # In real system: trigger thermal throttling, enable all cooling
```

### Form Factor and Integration

Jetson modules come in different form factors that affect integration:

**SO-DIMM modules** (Nano, TX2, Xavier NX) are compact and can be mounted on a carrier board like a laptop memory module. This is suitable for robots with moderate space constraints.

**MEZZanine modules** (AGX Orin) are larger and connect via high-speed connectors to custom carrier boards. This provides more I/O options and better thermal paths but requires more space.

When designing a robot, you must allocate space for the compute module, carrier board, power regulation, cooling, and I/O connections. The AGX Orin module plus carrier can occupy 100-200 cm² of board space.

### Recommended Configurations

Based on our analysis, here are recommended configurations for different robot scales:

**Lightweight Humanoid (under 20kg)**: Jetson Orin Nano with passive or small active cooling. Focus on efficient perception models and consider a secondary microcontroller for low-level motor control.

**Medium Humanoid (20-40kg)**: Jetson Orin NX or AGX Orin (15W mode). Active cooling recommended. Good balance of compute and power efficiency.

**Full-Scale Humanoid (40kg+)**: Jetson AGX Orin (30-60W mode). Robust active cooling system. Can run multiple perception models and LLM-based planning simultaneously.

**Development Platform**: Jetson AGX Orin Developer Kit for maximum flexibility. Connect to external GPU workstation for simulation. Migrate optimized models to production module for deployment.

## Development Workflow

A typical development workflow leverages both Jetson hardware and more powerful simulation workstations:

1. **Simulation Phase**: Develop and test algorithms on an RTX-equipped workstation running Isaac Sim. Train neural networks using PyTorch with CUDA acceleration.

2. **Transfer Phase**: Export trained models and convert them to TensorRT engines optimized for Jetson. Test on Jetson in the loop (hardware-in-the-loop) with the simulation.

3. **Deployment Phase**: Deploy optimized models to the robot's Jetson module. Collect real-world data for domain randomization and sim-to-real refinement.

This workflow allows you to leverage the massive compute of RTX GPUs for training while deploying efficient models to edge hardware.

```python
class DevelopmentWorkflow:
    """
    Manages the simulation-to-deployment workflow.
    """

    def __init__(self):
        self.stages = {
            'simulation': self._run_simulation,
            'optimization': self._optimize_models,
            'testing': self._test_on_hardware,
            'deployment': self._deploy_to_robot
        }

    def execute_phase(self, phase: str, config: dict) -> dict:
        """Execute a workflow phase."""
        if phase not in self.stages:
            raise ValueError(f"Unknown phase: {phase}")

        return self.stages[phase](config)

    def _run_simulation(self, config: dict) -> dict:
        """Run simulation training/evaluation."""
        return {
            'phase': 'simulation',
            'models_created': config.get('num_models', 1),
            'training_time_hours': config.get('training_hours', 24),
            'success_rate': 0.85,
            'output': 'models/simulated_policy.pt'
        }

    def _optimize_models(self, config: dict) -> dict:
        """Optimize models for edge deployment."""
        return {
            'phase': 'optimization',
            'input_model': config.get('input_model', 'models/simulated_policy.pt'),
            'output_engine': 'models/optimized_policy.engine',
            'original_size_mb': 256,
            'optimized_size_mb': 64,
            'speedup_factor': 4.5,
            'precision': 'fp16'
        }

    def _test_on_hardware(self, config: dict) -> dict:
        """Test on Jetson hardware in the loop."""
        return {
            'phase': 'hardware_testing',
            'target_device': config.get('device', 'jetson_agx_orin'),
            'test_duration_minutes': config.get('duration', 60),
            'inference_latency_ms': 12.5,
            'thermal_behavior': 'acceptable',
            'power_consumption_watts': 42
        }

    def _deploy_to_robot(self, config: dict) -> dict:
        """Deploy to robot and validate."""
        return {
            'phase': 'deployment',
            'robot': config.get('robot_id', 'humanoid_01'),
            'deployment_time': 'instant',
            'system_health': 'nominal',
            'ready_for_field_test': True
        }

    def run_full_pipeline(self, config: dict) -> dict:
        """Execute complete workflow pipeline."""
        results = {}
        for phase in self.stages.keys():
            results[phase] = self.execute_phase(phase, config)
        return results

# Example workflow execution
workflow = DevelopmentWorkflow()
config = {
    'num_models': 5,
    'training_hours': 48,
    'input_model': 'models/simulated_policy.pt',
    'device': 'jetson_agx_orin',
    'duration': 120,
    'robot_id': 'humanoid_alpha'
}

print("Running Development Pipeline...")
print("=" * 40)
results = workflow.run_full_pipeline(config)

for phase, result in results.items():
    print(f"\n{phase.upper()}:")
    for key, value in result.items():
        print(f"  {key}: {value}")
```

## Key Takeaways

Hardware selection is a fundamental constraint that shapes everything else in a Physical AI system. The NVIDIA Jetson platform provides a range of edge computing solutions optimized for power efficiency, while RTX GPUs provide the simulation compute necessary for development. Matching your hardware to your requirements—and understanding the trade-offs between performance, power, and cost—is essential for building successful humanoid robots.

The key principles to remember are:

- **Compute requirements are massive**: Modern Physical AI requires hundreds of TOPS for real-time performance.
- **Power is the limiting factor**: Mobile robots must balance compute against actuation demands.
- **Jetson provides a clear upgrade path**: Start with what you need, scale up as requirements grow.
- **Simulation and deployment hardware differ**: Develop on RTX, deploy on Jetson.

With these foundations established, we can now explore the software frameworks that bring these hardware platforms to life.
