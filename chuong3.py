import numpy as np
import cv2
from matplotlib import pyplot as plt

# Đọc ảnh gốc
image_path = r'E:\Daihoc\XuLyAnh\Image\24bit.png'  # Sử dụng raw string
original_image = cv2.imread(image_path)

# Kiểm tra xem ảnh có được đọc đúng không
if original_image is None:
    raise ValueError("Không thể đọc được file ảnh. Vui lòng kiểm tra lại đường dẫn.")

# Chuyển đổi ảnh từ BGR sang RGB (do OpenCV đọc ảnh dưới dạng BGR)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

# Thêm nhiễu Salt and Pepper vào ảnh
def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = image.copy()
    num_salt = np.ceil(salt_prob * image.size)
    num_pepper = np.ceil(pepper_prob * image.size)

    # Tạo nhiễu salt (các điểm trắng)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[coords[0], coords[1], :] = 1

    # Tạo nhiễu pepper (các điểm đen)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[coords[0], coords[1], :] = 0

    return noisy_image

salt_prob = 0.02
pepper_prob = 0.02
noisy_image = add_salt_and_pepper_noise(original_image, salt_prob, pepper_prob)

# Áp dụng bộ lọc trung vị
median_filtered_image = cv2.medianBlur(noisy_image, 3)

# Áp dụng bộ lọc trung bình
mean_filtered_image = cv2.blur(noisy_image, (3, 3))

# Áp dụng bộ lọc Gaussian
gaussian_filtered_image = cv2.GaussianBlur(noisy_image, (3, 3), 0)

# Hiển thị ảnh gốc, ảnh nhiễu, và các ảnh đã lọc
fig, axs = plt.subplots(1, 5, figsize=(20, 10))
axs[0].imshow(original_image)
axs[0].set_title("Ảnh gốc")
axs[0].axis('off')

axs[1].imshow(noisy_image)
axs[1].set_title("Ảnh nhiễu Salt and Pepper")
axs[1].axis('off')

axs[2].imshow(median_filtered_image)
axs[2].set_title("Bộ lọc trung vị")
axs[2].axis('off')

axs[3].imshow(mean_filtered_image)
axs[3].set_title("Bộ lọc trung bình")
axs[3].axis('off')

axs[4].imshow(gaussian_filtered_image)
axs[4].set_title("Bộ lọc Gaussian")
axs[4].axis('off')

plt.show()
