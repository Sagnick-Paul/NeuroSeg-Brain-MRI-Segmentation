import cv2
from inference import predict

# Put any MRI test image in the same folder
image = cv2.imread("test_mri.png", cv2.IMREAD_GRAYSCALE)

mask = predict(image)

print("Prediction shape:", mask.shape)
print("Predicted tumor pixels:", mask.sum())
