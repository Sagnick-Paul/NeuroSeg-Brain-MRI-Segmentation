import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import time
import segmentation_models_pytorch as smp

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="NeuroSeg AI",
    layout="wide",
    page_icon="🧠"
)

DEVICE = torch.device("cpu")
IMAGE_SIZE = 384

# =========================================================
# THEME
# =========================================================

st.markdown("""
<style>
body {
    background-color: #0E1117;
}
.metric-card {
    background-color: #111827;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=80)
    st.title("NeuroSeg AI")

    st.markdown("---")

    threshold = st.slider("Segmentation Threshold", 0.1, 0.9, 0.5, 0.05)
    overlay_alpha = st.slider("Overlay Transparency", 0.1, 1.0, 0.5, 0.1)
    show_gradcam = st.toggle("Enable Grad-CAM", True)

    st.markdown("---")
    st.caption("Model: UNet + EfficientNet-B3")
    st.caption("Version: v1.0")
    st.caption("Device: CPU")

# =========================================================
# MODEL LOAD
# =========================================================

@st.cache_resource
def load_model():
    model = smp.Unet(
        encoder_name="efficientnet-b3",
        encoder_weights=None,
        in_channels=1,
        classes=1
    )

    model.load_state_dict(
        torch.load("tumour_model_v1.pth", map_location="cpu")
    )

    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# =========================================================
# METRIC FUNCTIONS
# =========================================================

def dice_score(pred, gt):
    pred = pred.flatten()
    gt = gt.flatten()

    intersection = np.sum(pred * gt)
    return (2. * intersection + 1e-8) / (np.sum(pred) + np.sum(gt) + 1e-8)


def iou_score(pred, gt):
    pred = pred.flatten()
    gt = gt.flatten()

    intersection = np.sum(pred * gt)
    union = np.sum((pred + gt) > 0)

    return (intersection + 1e-8) / (union + 1e-8)


def sensitivity_score(pred, gt):
    pred = pred.flatten()
    gt = gt.flatten()

    TP = np.sum((pred == 1) & (gt == 1))
    FN = np.sum((pred == 0) & (gt == 1))

    return (TP + 1e-8) / (TP + FN + 1e-8)

# =========================================================
# TIFF MASK LOADER
# =========================================================

def load_tif_mask(file):
    mask = Image.open(file)
    mask = np.array(mask)
    mask = (mask > 0).astype(np.uint8)
    return mask

# =========================================================
# GRAD CAM
# =========================================================

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor):
        self.model.zero_grad()
        input_tensor.requires_grad = True

        output = self.model(input_tensor)
        prob_map = torch.sigmoid(output)

        mask = (prob_map > 0.5).float()
        score = (prob_map * mask).sum()

        score.backward()

        if self.gradients is None or self.activations is None:
            return np.zeros((IMAGE_SIZE, IMAGE_SIZE))

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)

        cam = F.relu(cam)
        cam = cam.squeeze().detach().cpu().numpy()

        cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

target_layer = model.decoder.blocks[-1]
grad_cam = GradCAM(model, target_layer)

# =========================================================
# PREPROCESS
# =========================================================

def preprocess(image):
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=0)

    return torch.tensor(image, dtype=torch.float32).to(DEVICE)

# =========================================================
# MAIN UI
# =========================================================

st.title("🧠 Brain Tumor Segmentation Dashboard")

uploaded_file = st.file_uploader(
    "Upload MRI Image (TIF supported)",
    type=["png", "jpg", "jpeg", "tif", "tiff"]
)
uploaded_gt = st.file_uploader("Upload Ground Truth Mask (TIF)", type=["tif", "tiff"])

def load_mri_image(file):
    image = Image.open(file)

    # Convert to grayscale
    if image.mode != "L":
        image = image.convert("L")

    return np.array(image)

if uploaded_file:

    image_np = load_mri_image(uploaded_file)

    input_tensor = preprocess(image_np)

    start_time = time.time()
    with torch.no_grad():
        output = model(input_tensor)
    inference_time = time.time() - start_time

    prob = torch.sigmoid(output)
    pred = prob.squeeze().cpu().numpy()

    mask = (pred > threshold).astype(np.uint8)

    mask_resized = cv2.resize(
        mask,
        (image_np.shape[1], image_np.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    # =====================================================
    # Ground Truth Metrics
    # =====================================================

    if uploaded_gt is not None:
        gt_mask = load_tif_mask(uploaded_gt)

        gt_mask = cv2.resize(
            gt_mask,
            (mask_resized.shape[1], mask_resized.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

        dice = dice_score(mask_resized, gt_mask)
        iou = iou_score(mask_resized, gt_mask)
        sensitivity = sensitivity_score(mask_resized, gt_mask)

    else:
        dice = iou = sensitivity = 0.0

    # =====================================================
    # Visualization Processing
    # =====================================================

    tumor_area = int(np.sum(mask_resized))
    tumor_percentage = (tumor_area / image_np.size) * 100

    if tumor_area > 0:
        confidence = float(pred[mask == 1].mean())
    else:
        confidence = 0.0

    overlay = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)

    red_mask = np.zeros_like(overlay)
    red_mask[:, :, 2] = mask_resized * 255

    overlay = cv2.addWeighted(overlay, 1, red_mask, overlay_alpha, 0)

    # Bounding Box
    contours, _ = cv2.findContours(
        mask_resized,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Centroid
    M = cv2.moments(mask_resized)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.circle(overlay, (cx, cy), 5, (255, 255, 0), -1)

    # =====================================================
    # Tabs
    # =====================================================

    tab1, tab2, tab3 = st.tabs([
        "🧠 Segmentation",
        "🔥 Explainability",
        "📊 Advanced Metrics"
    ])

    with tab1:
        col1, col2 = st.columns(2)

        col1.image(image_np, caption="Original MRI", use_container_width=True)
        col2.image(overlay, caption="Tumor Overlay", use_container_width=True)

        _, buffer = cv2.imencode(".png", overlay)

        st.download_button(
            "Download Segmentation",
            buffer.tobytes(),
            "segmentation.png",
            "image/png"
        )

    # =====================================================
    # GradCAM Visualization
    # =====================================================

    with tab2:

        blended = None

        if show_gradcam:

            cam = grad_cam.generate(input_tensor)

            cam_resized = cv2.resize(
                cam,
                (image_np.shape[1], image_np.shape[0])
            )

            cam_resized = np.clip(cam_resized, 0, 1)

            cam_uint8 = np.uint8(cam_resized * 255)

            heatmap = cv2.applyColorMap(
                cam_uint8,
                cv2.COLORMAP_JET
            )

            original_bgr = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)

            heatmap = cv2.resize(
                heatmap,
                (original_bgr.shape[1], original_bgr.shape[0])
            )

            blended = cv2.addWeighted(
                original_bgr,
                0.6,
                heatmap,
                0.4,
                0
            )

        col1, col2 = st.columns(2)

        if blended is not None:
            col1.image(blended, caption="Grad-CAM Explanation", use_container_width=True)

        col2.image(image_np, caption="Original MRI", use_container_width=True)

    # =====================================================
    # Metrics Display
    # =====================================================

    with tab3:

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Tumor Area (px)", f"{tumor_area:,}")
        col2.metric("Tumor %", f"{tumor_percentage:.2f}%")
        col3.metric("Dice Score", f"{dice:.4f}")
        col4.metric("IoU Score", f"{iou:.4f}")

        st.metric("Sensitivity (Recall)", f"{sensitivity:.4f}")

st.markdown("---")
st.caption("⚠️ For research purposes only. Not intended for clinical diagnosis.")