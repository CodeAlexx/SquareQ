"""
SQUARE-Q Slab I/O Module

Handles efficient loading and saving of quantized weights with:
- Safetensors format for weight storage
- JSON sidecar for metadata (quantization config, calibration)
- Streaming prefetch with pinned memory for fast H2D transfers
- CUDA stream management for overlapping compute and transfer

Reference: SQUARE-Q Phase 2.3 Specification
"""

import torch
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import threading
import queue

try:
    from safetensors import safe_open
    from safetensors.torch import save_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("Warning: safetensors not available. Install with: pip install safetensors")


@dataclass
class QuantizedLayerMetadata:
    """Metadata for a single quantized layer."""
    layer_name: str
    in_features: int
    out_features: int
    bit_width: int
    group_size: int
    outlier_pct: float
    has_bias: bool
    smoothing_scales: Optional[List[float]] = None  # From calibration
    optimal_alpha: Optional[float] = None  # From calibration


@dataclass
class QuantizedModelMetadata:
    """Complete metadata for a quantized model."""
    model_name: str
    quantization_version: str
    layers: Dict[str, QuantizedLayerMetadata]
    calibration_file: Optional[str] = None  # Path to calibration JSON

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'model_name': self.model_name,
            'quantization_version': self.quantization_version,
            'layers': {
                name: asdict(layer) for name, layer in self.layers.items()
            },
            'calibration_file': self.calibration_file,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QuantizedModelMetadata':
        """Load from dictionary."""
        layers = {
            name: QuantizedLayerMetadata(**layer_data)
            for name, layer_data in data['layers'].items()
        }
        return cls(
            model_name=data['model_name'],
            quantization_version=data['quantization_version'],
            layers=layers,
            calibration_file=data.get('calibration_file'),
        )


def save_quantized_model(
    weights: Dict[str, torch.Tensor],
    metadata: QuantizedModelMetadata,
    save_path: str,
    sidecar_path: Optional[str] = None,
):
    """
    Save quantized model weights and metadata.

    Saves weights to safetensors format and metadata to JSON sidecar.

    Args:
        weights: Dict mapping tensor names to weight tensors
        metadata: Model metadata
        save_path: Path to save weights (e.g., "model_quantized.safetensors")
        sidecar_path: Optional path for metadata JSON (default: save_path.json)

    Example:
        >>> weights = {
        >>>     "layer.0.weight": quantized_weight,
        >>>     "layer.0.scales": scales,
        >>> }
        >>> metadata = QuantizedModelMetadata(...)
        >>> save_quantized_model(weights, metadata, "model_w4.safetensors")
    """
    if not SAFETENSORS_AVAILABLE:
        raise ImportError("safetensors is required for save_quantized_model")

    # Ensure save directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Save weights to safetensors
    save_file(weights, save_path)
    print(f"Saved quantized weights to {save_path}")

    # Save metadata to JSON sidecar
    if sidecar_path is None:
        sidecar_path = str(Path(save_path).with_suffix('.json'))

    with open(sidecar_path, 'w') as f:
        json.dump(metadata.to_dict(), f, indent=2)
    print(f"Saved metadata to {sidecar_path}")


def load_quantized_model(
    load_path: str,
    sidecar_path: Optional[str] = None,
    device: str = 'cpu',
) -> tuple[Dict[str, torch.Tensor], QuantizedModelMetadata]:
    """
    Load quantized model weights and metadata.

    Loads from safetensors and JSON sidecar.

    Args:
        load_path: Path to safetensors file
        sidecar_path: Optional path to metadata JSON (default: load_path.json)
        device: Device to load tensors to

    Returns:
        Tuple of (weights dict, metadata)

    Example:
        >>> weights, metadata = load_quantized_model("model_w4.safetensors")
        >>> for name, tensor in weights.items():
        >>>     print(f"{name}: {tensor.shape}")
    """
    if not SAFETENSORS_AVAILABLE:
        raise ImportError("safetensors is required for load_quantized_model")

    # Load weights from safetensors
    weights = {}
    with safe_open(load_path, framework="pt", device=device) as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)

    print(f"Loaded {len(weights)} tensors from {load_path}")

    # Load metadata from JSON sidecar
    if sidecar_path is None:
        sidecar_path = str(Path(load_path).with_suffix('.json'))

    with open(sidecar_path, 'r') as f:
        metadata_dict = json.load(f)

    metadata = QuantizedModelMetadata.from_dict(metadata_dict)
    print(f"Loaded metadata for {len(metadata.layers)} layers")

    return weights, metadata


class WeightPrefetcher:
    """
    Asynchronous weight prefetcher with pinned memory and CUDA streams.

    Prefetches weights from CPU to GPU using pinned memory for fast transfers.
    Overlaps H2D transfers with compute using separate CUDA streams.

    Example:
        >>> prefetcher = WeightPrefetcher(horizon=2, device='cuda:0')
        >>> prefetcher.start()
        >>>
        >>> # Queue weights for prefetch
        >>> for name, weight_cpu in weights.items():
        >>>     prefetcher.enqueue(name, weight_cpu)
        >>>
        >>> # Get prefetched weights (blocks until available)
        >>> weight_gpu = prefetcher.get("layer.0.weight")
        >>>
        >>> prefetcher.stop()
    """

    def __init__(
        self,
        horizon: int = 2,
        device: str = 'cuda:0',
        stream_pool_size: int = 2,
    ):
        """
        Args:
            horizon: Number of tensors to prefetch ahead
            device: Target CUDA device
            stream_pool_size: Number of CUDA streams for transfers
        """
        self.horizon = horizon
        self.device = torch.device(device)
        self.stream_pool_size = stream_pool_size

        # Create CUDA streams for async transfers
        if self.device.type == 'cuda':
            self.streams = [torch.cuda.Stream(device=self.device) for _ in range(stream_pool_size)]
            self.stream_idx = 0
        else:
            self.streams = []

        # Queues for prefetch pipeline
        self.request_queue = queue.Queue(maxsize=horizon)
        self.ready_cache: Dict[str, torch.Tensor] = {}

        # Worker thread for async prefetch
        self.worker_thread: Optional[threading.Thread] = None
        self.running = False
        self.lock = threading.Lock()

    def start(self):
        """Start the prefetch worker thread."""
        if self.running:
            return

        self.running = True
        self.worker_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.worker_thread.start()
        print(f"WeightPrefetcher started on {self.device} with horizon={self.horizon}")

    def stop(self):
        """Stop the prefetch worker thread."""
        if not self.running:
            return

        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=1.0)
        print("WeightPrefetcher stopped")

    def enqueue(self, name: str, tensor_cpu: torch.Tensor):
        """
        Enqueue a tensor for prefetching.

        Args:
            name: Tensor name/identifier
            tensor_cpu: CPU tensor to prefetch to GPU
        """
        if not self.running:
            raise RuntimeError("Prefetcher not started. Call start() first.")

        self.request_queue.put((name, tensor_cpu))

    def get(self, name: str, timeout: Optional[float] = None) -> torch.Tensor:
        """
        Get a prefetched tensor (blocks until available).

        Args:
            name: Tensor name/identifier
            timeout: Optional timeout in seconds

        Returns:
            GPU tensor
        """
        # Wait for tensor to be ready
        start_time = None
        if timeout is not None:
            import time
            start_time = time.time()

        while True:
            with self.lock:
                if name in self.ready_cache:
                    tensor = self.ready_cache.pop(name)
                    return tensor

            # Check timeout
            if timeout is not None:
                import time
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Timeout waiting for tensor '{name}'")

            # Brief sleep to avoid busy waiting
            import time
            time.sleep(0.001)

    def _prefetch_worker(self):
        """Worker thread that prefetches tensors asynchronously."""
        while self.running:
            try:
                # Get next prefetch request (with timeout to check self.running)
                try:
                    name, tensor_cpu = self.request_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Allocate pinned memory for fast H2D transfer
                if self.device.type == 'cuda':
                    # Pin memory for faster transfer
                    if not tensor_cpu.is_pinned():
                        tensor_pinned = tensor_cpu.pin_memory()
                    else:
                        tensor_pinned = tensor_cpu

                    # Select stream for this transfer
                    stream = self.streams[self.stream_idx]
                    self.stream_idx = (self.stream_idx + 1) % self.stream_pool_size

                    # Async H2D transfer
                    with torch.cuda.stream(stream):
                        tensor_gpu = tensor_pinned.to(self.device, non_blocking=True)

                    # Synchronize stream to ensure transfer completes
                    stream.synchronize()
                else:
                    # For CPU device, just copy
                    tensor_gpu = tensor_cpu.to(self.device)

                # Add to ready cache
                with self.lock:
                    self.ready_cache[name] = tensor_gpu

            except Exception as e:
                print(f"Error in prefetch worker: {e}")
                import traceback
                traceback.print_exc()


def load_quantized_model_streaming(
    load_path: str,
    layer_names: Optional[List[str]] = None,
    device: str = 'cuda:0',
    prefetch_horizon: Optional[int] = None,
    stream_pool_size: Optional[int] = None,
) -> tuple[Dict[str, torch.Tensor], QuantizedModelMetadata, WeightPrefetcher]:
    """
    Load quantized model with streaming prefetch.

    Returns a prefetcher that can asynchronously load weights on demand.
    Automatically adjusts parameters for tiny models (<20MB).

    Args:
        load_path: Path to safetensors file
        layer_names: Optional list of specific layers to load
        device: Target device
        prefetch_horizon: Number of weights to prefetch ahead (default: from env or auto-adjust)
        stream_pool_size: Number of CUDA streams (default: from env or auto-adjust)

    Returns:
        Tuple of (initial_weights, metadata, prefetcher)

    Example:
        >>> weights, metadata, prefetcher = load_quantized_model_streaming(
        >>>     "model_w4.safetensors",
        >>>     device='cuda:0',
        >>> )
        >>>
        >>> # Enqueue layers for prefetch
        >>> for layer_name in layer_names:
        >>>     weight = weights_cpu[f"{layer_name}.weight"]
        >>>     prefetcher.enqueue(f"{layer_name}.weight", weight)
        >>>
        >>> # Get weights as needed (async)
        >>> weight_gpu = prefetcher.get("layer.0.weight")
    """
    # Load metadata
    sidecar_path = str(Path(load_path).with_suffix('.json'))
    with open(sidecar_path, 'r') as f:
        metadata_dict = json.load(f)
    metadata = QuantizedModelMetadata.from_dict(metadata_dict)

    # Load weights to CPU first
    weights_cpu = {}
    total_bytes = 0
    with safe_open(load_path, framework="pt", device='cpu') as f:
        keys = list(f.keys())
        if layer_names:
            # Filter to requested layers
            keys = [k for k in keys if any(ln in k for ln in layer_names)]

        for key in keys:
            tensor = f.get_tensor(key)
            weights_cpu[key] = tensor
            total_bytes += tensor.numel() * tensor.element_size()

    print(f"Loaded {len(weights_cpu)} tensors ({total_bytes / 1e6:.1f}MB) to CPU from {load_path}")

    # Use provided values or fall back to env/defaults
    horizon = prefetch_horizon if prefetch_horizon is not None else SQUAREQ_PREFETCH_HORIZON
    stream_pool = stream_pool_size if stream_pool_size is not None else SQUAREQ_STREAM_POOL

    # Keep track of requested settings for diagnostics
    requested_horizon = horizon
    requested_stream_pool = stream_pool

    # Adjust for tiny models
    params = adjust_for_tiny_model(total_bytes, horizon=horizon, stream_pool=stream_pool)

    # Create prefetcher with adjusted parameters
    prefetcher = WeightPrefetcher(
        horizon=params['horizon'],
        device=device,
        stream_pool_size=params['stream_pool'],
    )
    prefetcher.start()
    # Attach diagnostics for downstream reporting
    prefetcher.requested_horizon = requested_horizon
    prefetcher.effective_horizon = params['horizon']
    prefetcher.requested_stream_pool = requested_stream_pool
    prefetcher.effective_stream_pool = params['stream_pool']

    return weights_cpu, metadata, prefetcher
