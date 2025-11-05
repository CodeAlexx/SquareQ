from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import torch


@dataclass
class LayerRecord:
    """Metadata for a single quantized layer inside a slab."""

    name: str
    out_features: int
    in_features: int
    padded_in_features: int
    has_bias: bool

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "out": int(self.out_features),
            "inp": int(self.in_features),
            "padded_in": int(self.padded_in_features),
            "has_bias": bool(self.has_bias),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "LayerRecord":
        return cls(
            name=str(data["name"]),
            out_features=int(data["out"]),
            in_features=int(data["inp"]),
            padded_in_features=int(data.get("padded_in", data["inp"])),
            has_bias=bool(data.get("has_bias", False)),
        )


@dataclass
class SlabManifest:
    """Top level manifest describing the slab."""

    model_name: str
    quant_version: str
    layout: str
    pack_k: int
    layers: List[LayerRecord] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "model_name": self.model_name,
            "quant_version": self.quant_version,
            "layout": self.layout,
            "pack_k": int(self.pack_k),
            "layers": [layer.to_dict() for layer in self.layers],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SlabManifest":
        layers = [LayerRecord.from_dict(entry) for entry in data.get("layers", [])]
        return cls(
            model_name=str(data.get("model_name", "unknown")),
            quant_version=str(data.get("quant_version", "bp8")),
            layout=str(data.get("layout", "rowwise")),
            pack_k=int(data.get("pack_k", 64)),
            layers=layers,
        )

    def layer_names(self) -> Iterator[str]:
        for layer in self.layers:
            yield layer.name


@dataclass
class LayerData:
    """Quantized tensors for a layer."""

    qweight: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor
    bias: Optional[torch.Tensor] = None

    def to_dict(self) -> Dict[str, torch.Tensor]:
        payload = {
            "qweight": self.qweight,
            "scale": self.scale,
            "zero_point": self.zero_point,
        }
        if self.bias is not None:
            payload["bias"] = self.bias
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, torch.Tensor]) -> "LayerData":
        return cls(
            qweight=data["qweight"],
            scale=data["scale"],
            zero_point=data["zero_point"],
            bias=data.get("bias"),
        )


@dataclass
class SlabStorage:
    """In-memory slab storage."""

    manifest: SlabManifest
    layers: Dict[str, LayerData]

    def __post_init__(self) -> None:
        missing = set(self.manifest.layer_names()) - set(self.layers.keys())
        if missing:
            raise ValueError(f"Missing quant data for layers: {sorted(missing)}")

    def __contains__(self, name: str) -> bool:
        return name in self.layers

    def names(self) -> Iterable[str]:
        return self.layers.keys()

    def __getitem__(self, name: str) -> LayerData:
        return self.layers[name]


def save_slab(path: str | Path, storage: SlabStorage) -> None:
    """Persist a slab to disk using torch.save."""
    path = Path(path)
    serialized_layers = {
        name: layer.to_dict() for name, layer in storage.layers.items()
    }
    payload = {
        "manifest": storage.manifest.to_dict(),
        "layers": serialized_layers,
    }
    torch.save(payload, path)


def load_slab(path: str | Path) -> SlabStorage:
    """Load a slab previously saved with :func:`save_slab`."""
    path = Path(path)
    payload = torch.load(path, map_location="cpu")
    manifest = SlabManifest.from_dict(payload["manifest"])
    layer_dict: Dict[str, LayerData] = {
        name: LayerData.from_dict(entry) for name, entry in payload["layers"].items()
    }
    return SlabStorage(manifest=manifest, layers=layer_dict)
