"""Export OSNet-x0.25 (MSMT17 re-ID weights) to ONNX for onnxruntime CPU inference.

Usage: .venv/bin/python scripts/export_osnet.py models/osnet_x0_25_msmt17.pth models/osnet_x0_25_msmt17.onnx
Weights come from the torchreid model zoo (Google Drive, see README). Input is
1x3x256x128 RGB, ImageNet-normalised; output is a 512-d feature vector.
"""
import sys
from collections import OrderedDict

import torch

from followme.vision.osnet import osnet_x0_25


def main(src, dst):
    model = osnet_x0_25(num_classes=1000, pretrained=False)
    sd = torch.load(src, map_location="cpu", weights_only=False)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    clean = OrderedDict((k[7:] if k.startswith("module.") else k, v) for k, v in sd.items())
    msd = model.state_dict()
    matched = {k: v for k, v in clean.items() if k in msd and msd[k].shape == v.shape}
    missing = [k for k in msd if k not in matched and not k.startswith("classifier")]
    print(f"matched {len(matched)}/{len(msd)} tensors; unmatched non-classifier: {missing[:5]}")
    if missing:
        raise SystemExit("weights do not match osnet_x0_25 architecture")
    model.load_state_dict(matched, strict=False)
    model.eval()
    x = torch.randn(1, 3, 256, 128)
    torch.onnx.export(model, x, dst, input_names=["input"], output_names=["features"],
                      dynamic_axes={"input": {0: "batch"}, "features": {0: "batch"}},
                      opset_version=17, dynamo=False)
    print("exported", dst, "feature dim", model(x).shape)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
