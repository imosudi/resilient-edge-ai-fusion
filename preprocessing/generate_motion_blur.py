import cv2
import numpy as np

image = cv2.imread("input.jpg")

if image is None:
    raise FileNotFoundError("Input image not found")

kernel_size = 15

# Horizontal motion blur kernel
kernel = np.zeros((kernel_size, kernel_size))

kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)

kernel = kernel / kernel_size

# Apply blur
blurred = cv2.filter2D(image, -1, kernel)

cv2.imwrite("motion_blur.jpg", blurred)

print("Motion blur image generated")