"""Exports an upscaling model to ONNX at the fixed tile size textures.py uses, to run on any DirectX 12 GPU.

    python tools/export_model.py <model.safetensors> <out.onnx>

Needs torch (the CPU build is enough), spandrel, onnx and onnxscript. The pipeline uses 4x-PBRify_UpscalerV4 by Kim2091
(CC0): https://github.com/Kim2091/Kim2091-Models/releases/tag/4x-PBRify_UpscalerV4
"""
import sys

import spandrel
import torch

TILE = 256

if len(sys.argv) != 3:
    sys.exit(__doc__)
model = spandrel.ModelLoader(device="cpu").load_from_file(sys.argv[1]).model.eval()
with torch.no_grad():
    torch.onnx.export(model, (torch.rand(1, 3, TILE, TILE),), sys.argv[2], opset_version=18, input_names=["input"],
                      output_names=["output"], dynamo=False, do_constant_folding=True)
print("wrote", sys.argv[2])
