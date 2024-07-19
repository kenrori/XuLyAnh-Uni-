import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh dưới dạng grayscale
img = cv2.imread('E:/Daihoc/XuLyAnh/Image/gray2.png', 0)

# Biến đổi Fourier
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# Tạo bộ lọc thông thấp (Low-Pass Filter)
def create_lowpass_filter(shape, cutoff):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols, 2), np.float32)
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            if distance <= cutoff:
                mask[i, j] = 1
    return mask

cutoff = 30  # Tần số cắt
lowpass_filter = create_lowpass_filter(img.shape, cutoff)

# Áp dụng bộ lọc thông thấp trên phổ tần số
dft_shift_filtered = dft_shift * lowpass_filter

# Biến đổi ngược Fourier
f_ishift = np.fft.ifftshift(dft_shift_filtered)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

# Hiển thị ảnh gốc và ảnh sau khi lọc
plt.figure(figsize=(12, 6))
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Ảnh gốc'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_back, cmap='gray')
plt.title('Ảnh sau khi lọc thông thấp'), plt.xticks([]), plt.yticks([])
plt.show()
