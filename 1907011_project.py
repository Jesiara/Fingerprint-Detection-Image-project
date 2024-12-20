
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
import math


PI = math.pi
sigma_x = 1  
sigma_y = 1  

# Gaussian function
def gauss_func(x, y):
    coeff = 1 / (2 * PI * sigma_x * sigma_y)
    val = (((x ** 2) / (sigma_x ** 2)) + ((y ** 2) / (sigma_y ** 2))) / 2
    value = math.exp(-val)
    value = value * coeff
    return value

# Gaussian kernel
def gaussian_kernel(size):
    k_row, k_col = size, size
    gauss_kernel = np.zeros((k_row, k_col), dtype=np.float32)
    center = (size - 1) / 2

    for i in range(k_row):
        for j in range(k_col):
            x = i - center
            y = j - center
            gauss_kernel[i, j] = gauss_func(x, y)
    
    # Normalize the kernel
    gauss_kernel /= np.sum(gauss_kernel)
    
    return gauss_kernel

def convolve(image, kernel):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
    convolved_image = np.zeros_like(image)
    for i in range(image_height):
        for j in range(image_width):
            convolved_image[i, j] = np.sum(kernel * padded_image[i:i+kernel_height, j:j+kernel_width])
    return convolved_image

def adaptive_threshold(image, block_size, C):
    height, width = image.shape
    new_image = np.zeros_like(image)
    pad = block_size // 2
    padded_image = np.pad(image, pad, mode='constant', constant_values=0)
    for i in range(height):
        for j in range(width):
            local_region = padded_image[i:i+block_size, j:j+block_size]
            local_mean = np.mean(local_region)
            threshold = local_mean - C
            new_image[i, j] = 255 if image[i, j] > threshold else 0
    return new_image


def erosion(image, kernel):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=255)
    eroded_image = np.zeros_like(image)
    for i in range(image_height):
        for j in range(image_width):
            local_region = padded_image[i:i+kernel_height, j:j+kernel_width]
            eroded_image[i, j] = np.min(local_region[kernel == 1])
    return eroded_image

def dilation(image, kernel):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
    dilated_image = np.zeros_like(image)
    for i in range(image_height):
        for j in range(image_width):
            local_region = padded_image[i:i+kernel_height, j:j+kernel_width]
            dilated_image[i, j] = np.max(local_region[kernel == 1])
    return dilated_image

def thinning_iteration(image, iter):
    marker = np.zeros(image.shape, np.uint8)
    rows, cols = image.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            P2, P3, P4 = image[i-1, j], image[i-1, j+1], image[i, j+1]
            P5, P6, P7 = image[i+1, j+1], image[i+1, j], image[i+1, j-1]
            P8, P9 = image[i, j-1], image[i-1, j-1]
        
            A = (P2 == 0 and P3 == 1) + (P3 == 0 and P4 == 1) + \
            (P4 == 0 and P5 == 1) + (P5 == 0 and P6 == 1) + \
            (P6 == 0 and P7 == 1) + (P7 == 0 and P8 == 1) + \
            (P8 == 0 and P9 == 1) + (P9 == 0 and P2 == 1)
            B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
            m1 = P2 * P4 * P6 if iter == 0 else P2 * P4 * P8
            m2 = P4 * P6 * P8 if iter == 0 else P2 * P6 * P8
        
            if A == 1 and (2 <= B <= 6) and m1 == 0 and m2 == 0:
                marker[i, j] = 1
    image[marker == 1] = 0

def zhang_suen_thinning(image, max_iter=10):
    binary_image = image.copy() // 255
    prev = np.zeros(binary_image.shape, np.uint8)
    diff = True

    for _ in range(max_iter):
        if not diff:
            break
        thinning_iteration(binary_image, 0)
        thinning_iteration(binary_image, 1)
        diff = not np.array_equal(binary_image, prev)
        prev = binary_image.copy()

    return binary_image * 255

def minutiae_extraction(skel):
    minutiae_points = []
    for i in range(1, skel.shape[0] - 1):
        for j in range(1, skel.shape[1] - 1):
            if skel[i, j] == 255:
                neighbors = skel[i-1:i+2, j-1:j+2].flatten()
                transitions = np.count_nonzero(np.diff(neighbors == 255))
                if transitions == 2:
                    minutiae_points.append((i, j))  # Ridge ending
                elif transitions > 2:
                    minutiae_points.append((i, j))  # Bifurcation
    return minutiae_points

def match_minutiae_points(points1, points2, max_distance=10):
    matched_points = 0
    for p1 in points1:
        for p2 in points2:
            distance = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            if distance <= max_distance:
                matched_points += 1
                break
    return matched_points


fingerprint_image1 = cv2.imread('input_img.jpg', cv2.IMREAD_GRAYSCALE)
fingerprint_image2 = cv2.imread('input_img1.jpg', cv2.IMREAD_GRAYSCALE)

r1 = cv2.rotate(fingerprint_image2, cv2.ROTATE_90_CLOCKWISE)
r2 = cv2.rotate(r1, cv2.ROTATE_90_CLOCKWISE)
r3 = cv2.rotate(r2, cv2.ROTATE_90_CLOCKWISE)




gaussian_kernel = gaussian_kernel(5)
filtered_img1 = convolve(fingerprint_image1, gaussian_kernel)
filtered_img2 = convolve(fingerprint_image2, gaussian_kernel)
filtered_r1 = convolve(r1, gaussian_kernel)
filtered_r2 = convolve(r2, gaussian_kernel)
filtered_r3 = convolve(r2, gaussian_kernel)

block_size = 11
C = 2
thresh1 = adaptive_threshold(filtered_img1, block_size, C)

thresh2 = adaptive_threshold(filtered_img2, block_size, C)
thresh_r1 = adaptive_threshold(filtered_r1, block_size, C)
thresh_r2 = adaptive_threshold(filtered_r2, block_size, C)
thresh_r3 = adaptive_threshold(filtered_r3, block_size, C)



kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


dilated_img1 = dilation(thresh1, kernel)
dilated_img2 = dilation(thresh2, kernel)
dilated_r1 = dilation(thresh_r1, kernel)
dilated_r2 = dilation(thresh_r2, kernel)
dilated_r3 = dilation(thresh_r3, kernel)

closed_img1 = erosion(dilated_img1, kernel)
closed_img2 = erosion(dilated_img2, kernel)
closed_r1 = erosion(dilated_r1, kernel)
closed_r2 = erosion(dilated_r2, kernel)
closed_r3 = erosion(dilated_r3, kernel)

# Apply Zhang-Suen Thinning Algorithm
skeleton1 = zhang_suen_thinning(closed_img1)
skeleton2 = zhang_suen_thinning(closed_img2)
skeleton_r1 = zhang_suen_thinning(closed_r1)
skeleton_r2 = zhang_suen_thinning(closed_r2)
skeleton_r3 = zhang_suen_thinning(closed_r3)


# Minutiae Extraction
minutiae_points1 = minutiae_extraction(skeleton1)
minutiae_points2 = minutiae_extraction(skeleton2)
minutiae_pointsr1 = minutiae_extraction(skeleton_r1)
minutiae_pointsr2 = minutiae_extraction(skeleton_r2)
minutiae_pointsr3 = minutiae_extraction(skeleton_r3)


# Minutiae Matching
matched_points = match_minutiae_points(minutiae_points1, minutiae_points2)
total_points = len(minutiae_points1) + len(minutiae_points2)
match_percentage = (2 * matched_points / total_points) * 100

# Minutiae Matching
matched_pointsr1 = match_minutiae_points(minutiae_points1, minutiae_pointsr1)
total_pointsr1 = len(minutiae_points1) + len(minutiae_pointsr1)
match_percentager1 = (2 * matched_pointsr1 / total_pointsr1) * 100

# Minutiae Matching
matched_pointsr2 = match_minutiae_points(minutiae_points1, minutiae_pointsr2)
total_pointsr2 = len(minutiae_points1) + len(minutiae_pointsr2)
match_percentager2 = (2 * matched_pointsr2 / total_pointsr2) * 100

# Minutiae Matching
matched_pointsr3 = match_minutiae_points(minutiae_points1, minutiae_pointsr3)
total_pointsr3 = len(minutiae_points1) + len(minutiae_pointsr3)
match_percentager3 = (2 * matched_pointsr3 / total_pointsr3) * 100

match_per = []
match_per.append(match_percentage)
match_per.append(match_percentager1)
match_per.append(match_percentager2)
match_per.append(match_percentager3)

maxx=max(match_per)


print(f"Match Percentage: {maxx:.2f}%")

# Visualize Minutiae Points
skeleton_rgb1 = cv2.cvtColor(skeleton1, cv2.COLOR_GRAY2BGR)
skeleton_rgb2 = cv2.cvtColor(skeleton2, cv2.COLOR_GRAY2BGR)
for point in minutiae_points1:
    cv2.circle(skeleton_rgb1, point[::-1], 3, (0, 0, 255), -1)
for point in minutiae_points2:
    cv2.circle(skeleton_rgb2, point[::-1], 3, (0, 0, 255), -1)

cv2.imwrite('skeleton_minutiae1.jpg', skeleton_rgb1)
cv2.imwrite('skeleton_minutiae2.jpg', skeleton_rgb2)


cv2.imwrite('filtered_img1.jpg', filtered_img1)
cv2.imwrite('filtered_img2.jpg', filtered_img2)
cv2.imwrite('thresh1.jpg', thresh1)
cv2.imwrite('thresh2.jpg', thresh2)
cv2.imwrite('dilated_img1.jpg', dilated_img1)
cv2.imwrite('dilated_img2.jpg', dilated_img2)
cv2.imwrite('closed_img1.jpg', closed_img1)
cv2.imwrite('closed_img2.jpg', closed_img2)
cv2.imwrite('skeleton1.jpg', skeleton1)
cv2.imwrite('skeleton2.jpg', skeleton2)



# Define images and titles for plotting
images1 = [
    fingerprint_image1, filtered_img1, thresh1,  dilated_img1,  closed_img1, skeleton1, skeleton_rgb1
]
titles1 = [
    'Original Image 1', 'Filtered Image 1', 'Thresholded Image 1', 'Dilated Image 1', 'Closed Image 1', 'Skeleton Image 1', 'Skeleton with Minutiae 1'
]

images2 = [
    fingerprint_image2, filtered_img2, thresh2, dilated_img2, closed_img2, skeleton2, skeleton_rgb2
]
titles2 = [
    'Original Image 2', 'Filtered Image 2', 'Thresholded Image 2',  'Dilated Image 2', 'Closed Image 2', 'Skeleton Image 2', 'Skeleton with Minutiae 2'
]

def plot_images(images, titles, figure_title):
    fig, axes = plt.subplots(len(images) // 3 + (len(images) % 3 > 0), 3, figsize=(12, len(images) * 2))
    fig.suptitle(figure_title, fontsize=26)  

    for ax in axes.flat:
        ax.tick_params(axis='both', which='both', labelsize=12)  

    for i, ax in enumerate(axes.flat):
        if i < len(images):
            if len(images[i].shape) == 2:
                ax.imshow(images[i], cmap='gray')
            else:  
                ax.imshow(images[i])
            ax.set_title(titles[i], fontsize=22)
            ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


# Plot images for both fingerprint images
plot_images(images1, titles1, "Processing Steps for Fingerprint Image 1")
plot_images(images2, titles2, "Processing Steps for Fingerprint Image 2")

cv2.imshow('Skeleton with Minutiae Points 1', skeleton_rgb1)
cv2.imshow('Skeleton with Minutiae Points 2', skeleton_rgb2)



cv2.waitKey(0)
cv2.destroyAllWindows()

