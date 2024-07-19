import cv2
import numpy as np
import matplotlib.pyplot as plt

# Bước 1: Đọc ảnh dưới dạng grayscale
img = cv2.imread('E:/Daihoc/XuLyAnh/Image/gray2.png', 0)

# Bước 2: Thực hiện biến đổi Fourier
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)

# Bước 3: Dịch chuyển phổ tần số
dft_shift = np.fft.fftshift(dft)

# Bước 4: Tính phổ biên độ
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

# Bước 5: Hiển thị ảnh gốc và phổ tần số
plt.figure(figsize=(12, 6))
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Ảnh gốc'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Phổ tần số'), plt.xticks([]), plt.yticks([])
plt.show()
