import cv2

image = cv2.imread("input.jpg")

if image is None:
    raise FileNotFoundError("Input image not found")

# Simulate low-light
dark = (image * 0.3).astype("uint8")

cv2.imwrite("low_light.jpg", dark)

print("Low-light image generated")