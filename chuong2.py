import cv2
import numpy as np
from matplotlib import pyplot as plt

def histogram_matching(source, template):
    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    s_quantiles = np.cumsum(s_counts).astype(np.float64) / source.size
    t_quantiles = np.cumsum(t_counts).astype(np.float64) / template.size

    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape).astype(np.uint8)

def histogram_matching_color(source, template):
    matched_channels = []
    for i in range(3):
        matched_channels.append(histogram_matching(source[:, :, i], template[:, :, i]))
    return cv2.merge(matched_channels)

def process_image(image_gray_path, template_gray_path, image_color_path, template_color_path):
    # Đọc ảnh xám
    image_gray = cv2.imread(image_gray_path, cv2.IMREAD_GRAYSCALE)
    if image_gray is None:
        print(f"Không thể đọc ảnh từ đường dẫn {image_gray_path}")
        return
    
    # Cân bằng lược đồ mức xám
    equalized_gray = cv2.equalizeHist(image_gray)

    # Đọc ảnh màu
    image_color = cv2.imread(image_color_path)
    if image_color is None:
        print(f"Không thể đọc ảnh từ đường dẫn {image_color_path}")
        return

    # Cân bằng lược đồ mức xám cho ảnh màu
    ycrcb_image = cv2.cvtColor(image_color, cv2.COLOR_BGR2YCrCb)
    ycrcb_image[:, :, 0] = cv2.equalizeHist(ycrcb_image[:, :, 0])
    equalized_color = cv2.cvtColor(ycrcb_image, cv2.COLOR_YCrCb2BGR)

    # Hiển thị kết quả
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 3, 1)
    plt.title('Ảnh Xám Gốc')
    plt.imshow(image_gray, cmap='gray')

    plt.subplot(2, 3, 2)
    plt.title('Ảnh Cân Bằng Lược Đồ Mức Xám')
    plt.imshow(equalized_gray, cmap='gray')

    plt.subplot(2, 3, 4)
    plt.title('Ảnh Màu Gốc')
    plt.imshow(cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB))

    plt.subplot(2, 3, 5)
    plt.title('Ảnh Màu Cân Bằng Lược Đồ')
    plt.imshow(cv2.cvtColor(equalized_color, cv2.COLOR_BGR2RGB))

    # Nếu có ảnh template, thực hiện đối sánh lược đồ
    if template_gray_path and template_color_path:
        template_gray = cv2.imread(template_gray_path, cv2.IMREAD_GRAYSCALE)
        if template_gray is None:
            print(f"Không thể đọc ảnh từ đường dẫn {template_gray_path}")
            return
        matched_gray = histogram_matching(image_gray, template_gray)

        template_color = cv2.imread(template_color_path)
        if template_color is None:
            print(f"Không thể đọc ảnh từ đường dẫn {template_color_path}")
            return
        matched_color = histogram_matching_color(image_color, template_color)

        plt.subplot(2, 3, 3)
        plt.title('Ảnh Xám Đối Sánh')
        plt.imshow(matched_gray, cmap='gray')

        plt.subplot(2, 3, 6)
        plt.title('Ảnh Màu Đối Sánh')
        plt.imshow(cv2.cvtColor(matched_color, cv2.COLOR_BGR2RGB))

    plt.tight_layout()
    plt.show()

# Đường dẫn đến ảnh và template
image_gray_path = 'E:/Daihoc/XuLyAnh/Image/gray.png'
template_gray_path = 'E:/Daihoc/XuLyAnh/Image/gray2.png'
image_color_path = 'E:/Daihoc/XuLyAnh/Image/24bit.png'
template_color_path = 'E:/Daihoc/XuLyAnh/Image/24bit-3.png'

# Gọi hàm xử lý ảnh
process_image(image_gray_path, template_gray_path, image_color_path, template_color_path)
