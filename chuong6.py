import cv2
import numpy as np
import matplotlib.pyplot as plt

def butterworth_lowpass_filter(shape, D0, n): #thông thấp
    P, Q = shape
    H = np.zeros((P, Q))
    for u in range(P):
        for v in range(Q):
            D = np.sqrt((u - P / 2) ** 2 + (v - Q / 2) ** 2)
            H[u, v] = 1 / (1 + (D / D0) ** (2 * n))
    return H

def butterworth_highpass_filter(shape, D0, n): #thông cao
    P, Q = shape
    H = np.zeros((P, Q))
    for u in range(P):
        for v in range(Q):
            D = np.sqrt((u - P / 2) ** 2 + (v - Q / 2) ** 2)
            if D == 0:
                H[u, v] = 0  # Tránh chia cho 0 bằng cách gán giá trị 0 cho trung tâm
            else:
                H[u, v] = 1 / (1 + (D0 / D) ** (2 * n))
    return H

def apply_filter(img, H):
    # Thực hiện biến đổi Fourier
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    # Áp dụng bộ lọc Butterworth trên phổ tần số
    dft_shift_filtered = dft_shift.copy()
    dft_shift_filtered[:, :, 0] *= H
    dft_shift_filtered[:, :, 1] *= H
    
    # Biến đổi ngược Fourier để có ảnh sau khi lọc
    f_ishift = np.fft.ifftshift(dft_shift_filtered)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    
    return img_back

# Đọc ảnh dưới dạng grayscale
img = cv2.imread('E:/Daihoc/XuLyAnh/Image/gray2.png', 0)

# Tham số của bộ lọc Butterworth
D0 = 30  # Tần số cắt
n = 2    # Bậc của bộ lọc

# Tạo bộ lọc Butterworth thông thấp và thông cao
H_low = butterworth_lowpass_filter(img.shape, D0, n)
H_high = butterworth_highpass_filter(img.shape, D0, n)

# Áp dụng bộ lọc Butterworth
img_low_filtered = apply_filter(img, H_low)
img_high_filtered = apply_filter(img, H_high)

# Hiển thị ảnh gốc và ảnh sau khi lọc Butterworth
plt.figure(figsize=(18, 6))
plt.subplot(131), plt.imshow(img, cmap='gray')
plt.title('Ảnh gốc'), plt.xticks([]), plt.yticks([])

plt.subplot(132), plt.imshow(img_low_filtered, cmap='gray')
plt.title('Ảnh sau khi lọc Butterworth thông thấp'), plt.xticks([]), plt.yticks([])

plt.subplot(133), plt.imshow(img_high_filtered, cmap='gray')
plt.title('Ảnh sau khi lọc Butterworth thông cao'), plt.xticks([]), plt.yticks([])

plt.show()
