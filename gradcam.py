"""
Grad-CAM for FakeWave's DeepfakeCNN, computed via plain PyTorch forward/
backward hooks (no captum/lime/shap).

Hooks the model's actual last conv layer (model.features[-1][0], the
Conv2d(64, 128) in the final block) on whatever model instance is passed in
-- callers must pass the trained detector.pt weights (via predict.load_model),
never a fresh/pretrained one, or the heatmap is meaningless noise.

The heatmap is backpropped from the PREDICTED class's score: since
DeepfakeCNN emits a single fake-vs-real logit (not two class logits), "the
predicted class's score" is the raw logit when the prediction is "fake"
(higher logit = more fake) and the negated logit when the prediction is
"real" (higher = more real). That mirrors the two-logit Grad-CAM case
without needing a second output head.
"""

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F


def _target_layer(model):
    return model.features[-1][0]  # Conv2d of the final conv block


class GradCAM:
    """Forward/backward-hook Grad-CAM over a single target conv layer."""

    def __init__(self, model, target_layer=None):
        self.model = model
        self.layer = target_layer if target_layer is not None else _target_layer(model)
        self._activations = None
        self._gradients = None
        self._fwd_handle = self.layer.register_forward_hook(self._save_activations)
        self._bwd_handle = self.layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inp, out):
        self._activations = out.detach()

    def _save_gradients(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def remove(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __call__(self, logmel_batch):
        """logmel_batch: (1, 1, n_mels, time) tensor.

        Returns (cam, label, confidence, probs): cam is an (n_mels, time)
        numpy array in [0, 1], upsampled to the input spectrogram's own
        dimensions; label/confidence/probs match predict.predict()'s return.
        """
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        logit = self.model(logmel_batch)  # (1,) raw logit, higher = more "fake"
        fake_prob = torch.sigmoid(logit).item()
        label = "fake" if fake_prob > 0.5 else "real"
        probs = {"real": 1.0 - fake_prob, "fake": fake_prob}

        score = logit if label == "fake" else -logit
        score.backward()

        activations = self._activations[0]  # (C, H, W)
        gradients = self._gradients[0]  # (C, H, W)
        weights = gradients.mean(dim=(1, 2))  # (C,) global-average-pooled gradients
        cam = torch.einsum("c,chw->hw", weights, activations)
        cam = F.relu(cam)

        target_h, target_w = logmel_batch.shape[-2], logmel_batch.shape[-1]
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0), size=(target_h, target_w),
            mode="bilinear", align_corners=False,
        ).squeeze().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, label, probs[label], probs


def render_spectrogram_image(logmel):
    """logmel: (1, n_mels, time) dB-scaled log-mel tensor, as returned by
    preprocess.preprocess_clip(). Returns an (n_mels, time, 3) uint8 RGB
    array with low mel bins at the bottom row (matching preprocess.py's own
    imshow(..., origin="lower") convention)."""
    arr = logmel.squeeze(0).cpu().numpy()
    arr = np.flipud(arr)
    normed = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    rgb = matplotlib.colormaps["magma"](normed)[..., :3]
    return (rgb * 255).astype(np.uint8)


def overlay_cam_on_spectrogram(spectrogram_rgb, cam, alpha=0.45):
    """spectrogram_rgb: (H, W, 3) uint8, already bottom-up oriented (see
    render_spectrogram_image). cam: (H, W) float array in [0, 1], same
    (unflipped) orientation as the log-mel tensor it was computed from.
    Returns an (H, W, 3) uint8 blend -- the spectrogram stays visible
    underneath a jet-colormap heatmap."""
    cam_flipped = np.flipud(cam)
    heat = (matplotlib.colormaps["jet"](cam_flipped)[..., :3] * 255).astype(np.uint8)
    blended = (
        spectrogram_rgb.astype(np.float32) * (1 - alpha)
        + heat.astype(np.float32) * alpha
    ).astype(np.uint8)
    return blended


def explain(path, model, config=None):
    """Full pipeline for the Gradio app: raw audio file -> verdict + both
    images. `model` must be the actual trained detector.pt (see module
    docstring) -- this function does no loading of its own so callers can't
    accidentally point it at fresh/pretrained weights.

    Returns (label, confidence, probs, spectrogram_rgb, overlay_rgb).
    """
    from preprocess import CONFIG, preprocess_clip

    config = config or CONFIG
    logmel = preprocess_clip(path, config)  # (1, n_mels, time)
    batch = logmel.unsqueeze(0)  # (1, 1, n_mels, time)

    cam_engine = GradCAM(model)
    try:
        cam, label, confidence, probs = cam_engine(batch)
    finally:
        cam_engine.remove()

    spectrogram_rgb = render_spectrogram_image(logmel)
    overlay_rgb = overlay_cam_on_spectrogram(spectrogram_rgb, cam)
    return label, confidence, probs, spectrogram_rgb, overlay_rgb
