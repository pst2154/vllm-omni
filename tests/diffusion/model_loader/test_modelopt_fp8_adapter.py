# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.model_loader.checkpoint_adapters import (
    ModelOptFp8CheckpointAdapter,
    ModelOptNvFp4CheckpointAdapter,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _PackedModelOptModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.block = nn.Module()
        self.transformer.block.to_qkv = nn.Linear(2, 2, bias=False)


class _QuantizedPackedModelOptModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.block = nn.Module()
        self.transformer.block.to_qkv = nn.Module()
        self.transformer.block.to_qkv.register_parameter(
            "weight",
            nn.Parameter(torch.empty(2, 2, dtype=torch.float8_e4m3fn), requires_grad=False),
        )
        self.transformer.block.to_qkv.register_parameter(
            "weight_scale",
            nn.Parameter(torch.empty(1), requires_grad=False),
        )
        self.transformer.block.to_qkv.register_parameter(
            "input_scale",
            nn.Parameter(torch.empty(1), requires_grad=False),
        )


def _make_source() -> SimpleNamespace:
    return SimpleNamespace(
        subfolder="transformer",
        prefix="transformer.",
    )


def test_modelopt_adapter_dequantizes_fp8_weight_for_full_precision_target():
    model = _PackedModelOptModel()
    adapter = ModelOptFp8CheckpointAdapter(model, _make_source())
    fp8_weight = torch.tensor([[2.0, -4.0], [1.0, 3.0]], dtype=torch.float32).to(torch.float8_e4m3fn)
    scale = torch.tensor([0.5], dtype=torch.float32)

    adapted = list(
        adapter.adapt(
            iter(
                [
                    ("transformer.block.to_q.weight_scale", scale),
                    ("transformer.block.to_q.input_scale", torch.tensor([1.0])),
                    ("transformer.block.to_q.weight", fp8_weight),
                ]
            )
        )
    )

    assert [name for name, _ in adapted] == ["transformer.block.to_q.weight"]
    assert adapted[0][1].dtype == model.transformer.block.to_qkv.weight.dtype
    assert torch.allclose(adapted[0][1], fp8_weight.to(torch.float32) * scale)


def test_modelopt_adapter_keeps_scale_tensors_for_quantized_target():
    model = _QuantizedPackedModelOptModel()
    adapter = ModelOptFp8CheckpointAdapter(model, _make_source())
    scale = torch.tensor([0.5], dtype=torch.float32)

    adapted = list(
        adapter.adapt(
            iter(
                [
                    ("transformer.block.to_q.weight_scale", scale),
                    ("transformer.block.to_q.input_scale", torch.tensor([1.0])),
                ]
            )
        )
    )

    assert [name for name, _ in adapted] == [
        "transformer.block.to_q.weight_scale",
        "transformer.block.to_q.input_scale",
    ]


def test_modelopt_nvfp4_adapter_keeps_scales_for_downstream_remap():
    adapter = ModelOptNvFp4CheckpointAdapter(nn.Module(), _make_source())
    prefix = "transformer.layers.0.self_attn.add_q_proj"
    checkpoint_tensors = [
        (f"{prefix}.input_scale", torch.tensor([1.0])),
        (f"{prefix}.weight_scale", torch.tensor([0.5])),
        (f"{prefix}.weight_scale_2", torch.tensor([0.25])),
        (f"{prefix}.weight", torch.tensor([[1]], dtype=torch.uint8)),
    ]

    adapted = list(adapter.adapt(iter(checkpoint_tensors)))

    assert [name for name, _ in adapted] == [name for name, _ in checkpoint_tensors]
