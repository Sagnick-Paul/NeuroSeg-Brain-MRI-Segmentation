# import torch
# import torch.nn.functional as F
# import segmentation_models_pytorch as smp
# import cv2
# import numpy as np
# import os

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# IMAGE_SIZE = 256

# # ── Locate model weights ──────────────────────────────────────────────────────
# weights_path = "tumour_model_v1.pth"
# if not os.path.exists(weights_path):
#     weights_path = os.path.join("backend", "model", "tumour_model_v1.pth")

# if not os.path.exists(weights_path):
#     raise FileNotFoundError(
#         f"Model weights not found at 'tumour_model_v1.pth' or '{weights_path}'"
#     )

# # ── Model: UNet++ with EfficientNet-B3 backbone ───────────────────────────────
# # The saved checkpoint was trained with UnetPlusPlus + in_channels=3 (RGB MRI).
# model = smp.UnetPlusPlus(
#     encoder_name="efficientnet-b3",
#     encoder_weights=None,
#     in_channels=3,
#     classes=1,
# )

# model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
# model.to(DEVICE)
# model.eval()


# # ── Grad-CAM ──────────────────────────────────────────────────────────────────
# class GradCAM:
#     """Gradient-weighted Class Activation Mapping for UNet++ decoder."""

#     def __init__(self, model, target_layer):
#         self.model = model
#         self.gradients = None
#         self.activations = None

#         target_layer.register_forward_hook(self._save_activation)
#         target_layer.register_full_backward_hook(self._save_gradient)

#     def _save_activation(self, module, input, output):
#         self.activations = output

#     def _save_gradient(self, module, grad_input, grad_output):
#         self.gradients = grad_output[0]

#     def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
#         self.model.zero_grad()
#         inp = input_tensor.clone().detach().requires_grad_(True)

#         output = self.model(inp)
#         prob_map = torch.sigmoid(output)

#         score = (prob_map * (prob_map > 0.5).float()).sum()
#         score.backward()

#         if self.gradients is None or self.activations is None:
#             return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

#         weights = self.gradients.mean(dim=(2, 3), keepdim=True)
#         cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
#         cam = cam.squeeze().detach().cpu().numpy()

#         if cam.ndim == 0:
#             return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

#         cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))
#         cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
#         return cam.astype(np.float32)


# # UNet++ final dense block for UNet++ is 'x_0_4'
# target_layer = model.decoder.blocks["x_0_4"]
# grad_cam = GradCAM(model, target_layer)


# # ── Preprocessing ─────────────────────────────────────────────────────────────
# def preprocess(image: np.ndarray) -> torch.Tensor:
#     """
#     Accepts a 2-D grayscale ndarray (H x W).
#     Resize -> convert to 3-channel RGB -> z-score normalize -> return (1, 3, H, W) tensor.
#     """
#     # 1. Spatial alignment
#     resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)

#     # 2. Channel reconstruction
#     rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)        # (H, W, 3)
#     # 3. Base scaling
#     normalised = rgb.astype(np.float32) / 255.0
#     # 4. Exact ImageNet Z-Score Normalization
#     mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
#     std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
#     normalised = (normalised - mean) / std
    
#     # 5. Tensor packaging
#     tensor = torch.from_numpy(normalised.transpose(2, 0, 1)).unsqueeze(0)
#     return tensor.to(DEVICE)


# # ── Inference ─────────────────────────────────────────────────────────────────
# def predict(image: np.ndarray) -> np.ndarray:
#     """Return binary segmentation mask (numpy, same shape as model output)."""
#     image_tensor = preprocess(image)
#     with torch.no_grad():
#         output = model(image_tensor)
#         prob = torch.sigmoid(output)
#         mask = (prob > 0.1).float()
#     return mask.cpu().numpy()


import os
import urllib.request
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import cv2
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# FIXED: Reverted from 384 to 256 to match your training set image size
IMAGE_SIZE = 256

# ── Automated Weights Download Configuration ──────────────────────────────────
WEIGHTS_DIR = os.path.join("backend", "model")
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "tumour_model_v1.pth")

def ensure_weights_exist():
    """Checks for local presence of weights; downloads from GitHub Release if missing."""
    if not os.path.exists(WEIGHTS_DIR):
        os.makedirs(WEIGHTS_DIR)
        
    if not os.path.exists(WEIGHTS_PATH):
        # !! REPLACE WITH YOUR ACTUAL GITHUB USERNAME AND REPO NAME LATER !!
        DOWNLOAD_URL = "https://github.com/Sagnick-Paul/NeuroSeg-Brain-MRI-Segmentation/releases/download/v1.0.0/tumour_model_v1.pth"
        print(f"Downloading model weights to {WEIGHTS_PATH}...")
        try:
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(DOWNLOAD_URL, WEIGHTS_PATH)
            print("✅ Model weights downloaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to automatically download weights: {str(e)}")

# Execute check before instantiating the model
ensure_weights_exist()

# ── Model Instantiation ───────────────────────────────────────────────────────
model = smp.UnetPlusPlus(
    encoder_name="efficientnet-b3",
    encoder_weights=None,
    in_channels=3,
    classes=1,
)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ── Grad-CAM Logic ────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        self.model.zero_grad()
        inp = input_tensor.clone().detach().requires_grad_(True)
        output = self.model(inp)
        prob_map = torch.sigmoid(output)
        score = (prob_map * (prob_map > 0.5).float()).sum()
        score.backward()

        if self.gradients is None or self.activations is None:
            return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = cam.squeeze().detach().cpu().numpy()

        if cam.ndim == 0:
            return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

        cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.astype(np.float32)

target_layer = model.decoder.blocks["x_0_4"]
grad_cam = GradCAM(model, target_layer)

# ── Fixed Preprocessing ───────────────────────────────────────────────────────
def preprocess(image: np.ndarray) -> torch.Tensor:
    """Matches notebook validation transformations exactly."""
    resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    
    # Base scaling [0, 1]
    normalised = rgb.astype(np.float32) / 255.0
    
    # FIXED: Added required ImageNet Z-Score Normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalised = (normalised - mean) / std
    
    tensor = torch.from_numpy(normalised.transpose(2, 0, 1)).unsqueeze(0)
    return tensor.to(DEVICE)

def predict(image: np.ndarray) -> np.ndarray:
    image_tensor = preprocess(image)
    with torch.no_grad():
        output = model(image_tensor)
        prob = torch.sigmoid(output)
        mask = (prob > 0.5).float() # Fixed threshold to standard 0.5 evaluation
    return mask.cpu().numpy()