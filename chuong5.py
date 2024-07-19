import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh dưới dạng grayscale
img = cv2.imread('E:/Daihoc/XuLyAnh/Image/gray2.png', 0)

# Bộ lọc thông cao
def high_pass_filter(img):
    kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(img, -1, kernel)

# Bộ lọc Laplacian
def laplacian_filter(img):
    return cv2.Laplacian(img, cv2.CV_64F)

# Bộ lọc Unsharp Masking
def unsharp_masking(img, sigma=1.0, strength=1.5):
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1 + strength, blurred, -strength, 0)

# Áp dụng các bộ lọc
img_high_pass = high_pass_filter(img)
img_laplacian = laplacian_filter(img)
img_unsharp = unsharp_masking(img)

# Hiển thị kết quả
plt.figure(figsize=(12, 12))
plt.subplot(221), plt.imshow(img, cmap='gray')
plt.title('Ảnh gốc'), plt.xticks([]), plt.yticks([])

plt.subplot(222), plt.imshow(img_high_pass, cmap='gray')
plt.title('Ảnh sau khi lọc thông cao'), plt.xticks([]), plt.yticks([])

plt.subplot(223), plt.imshow(img_laplacian, cmap='gray')
plt.title('Ảnh sau khi lọc Laplacian'), plt.xticks([]), plt.yticks([])

plt.subplot(224), plt.imshow(img_unsharp, cmap='gray')
plt.title('Ảnh sau khi Unsharp Masking'), plt.xticks([]), plt.yticks([])

plt.show()
