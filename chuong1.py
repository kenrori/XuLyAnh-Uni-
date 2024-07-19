import cv2
import numpy as np
from matplotlib import pyplot as plt

def adjust_brightness(image, value):
    """Thay đổi mức sáng của ảnh."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v = np.clip(v, 0, 255)
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def adjust_contrast(image, alpha):
    """Thay đổi độ tương phản của ảnh."""
    img = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    return img

def negative_image(image):
    """Tạo ảnh âm bản."""
    img = cv2.bitwise_not(image)
    return img

def threshold_image(image, thresh_value):
    """Phân ngưỡng ảnh."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY)
    return binary

def process_image(image_path):
    # Đọc ảnh màu
    image = cv2.imread(image_path)
    if image is None:
        print(f"Không thể đọc ảnh từ đường dẫn {image_path}")
        return
    
    # Thay đổi mức sáng
    bright_image = adjust_brightness(image, 50)
    dark_image = adjust_brightness(image, -50)

    # Thay đổi độ tương phản
    high_contrast_image = adjust_contrast(image, 2.0)
    low_contrast_image = adjust_contrast(image, 0.5)

    # Tạo ảnh âm bản
    negative = negative_image(image)

    # Phân ngưỡng ảnh
    binary_image = threshold_image(image, 128)

    # Hiển thị kết quả
    plt.figure(figsize=(12, 12))

    plt.subplot(3, 3, 1)
    plt.title('Ảnh Gốc')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    plt.subplot(3, 3, 2)
    plt.title('Ảnh Tăng Sáng')
    plt.imshow(cv2.cvtColor(bright_image, cv2.COLOR_BGR2RGB))

    plt.subplot(3, 3, 3)
    plt.title('Ảnh Giảm Sáng')
    plt.imshow(cv2.cvtColor(dark_image, cv2.COLOR_BGR2RGB))

    plt.subplot(3, 3, 4)
    plt.title('Ảnh Tăng Tương Phản')
    plt.imshow(cv2.cvtColor(high_contrast_image, cv2.COLOR_BGR2RGB))

    plt.subplot(3, 3, 5)
    plt.title('Ảnh Giảm Tương Phản')
    plt.imshow(cv2.cvtColor(low_contrast_image, cv2.COLOR_BGR2RGB))

    plt.subplot(3, 3, 6)
    plt.title('Ảnh Âm Bản')
    plt.imshow(cv2.cvtColor(negative, cv2.COLOR_BGR2RGB))

    plt.subplot(3, 3, 7)
    plt.title('Ảnh Phân Ngưỡng')
    plt.imshow(binary_image, cmap='gray')

    plt.tight_layout()
    plt.show()

# Đường dẫn đến ảnh
image_path = 'E:/Daihoc/XuLyAnh/Image/24bit.png'

# Gọi hàm xử lý ảnh
process_image(image_path)
