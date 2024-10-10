import cv2
import matplotlib.pyplot as plt

# Step 1: Load a grayscale image
image_path = 'grayscale.jpg'
gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Step 2: Apply histogram equalization
equalized_image = cv2.equalizeHist(gray_image)

# Step 3: Plot the original and equalized images
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(gray_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Equalized Image")
plt.imshow(equalized_image, cmap='gray')
plt.axis('off')

plt.show()

# Step 4: Plot histograms for both images
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Histogram of Original Image")
plt.hist(gray_image.ravel(), 256, [0, 256], color='blue')
plt.xlim([0, 256])

plt.subplot(1, 2, 2)
plt.title("Histogram of Equalized Image")
plt.hist(equalized_image.ravel(), 256, [0, 256], color='red')
plt.xlim([0, 256])

plt.show()