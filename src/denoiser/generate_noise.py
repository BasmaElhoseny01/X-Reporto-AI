import numpy as np
import cv2
import matplotlib.pyplot as plt

# Block-Pixels: where each pixel is set to zero with probability p
def add_block_pixel_noise(image, probability=0.05):
    """
    Add Block-Pixel noise to an image.
    
    Args:
        image: Input image (numpy array).
        probability: Probability of setting each pixel to zero.
    
    Returns:
        Noisy image (numpy array).
    """
    noisy_image = np.copy(image)
    mask = np.random.rand(*image.shape[:2]) < probability
    noisy_image[mask] = 0
    return noisy_image,image

# Convolve-Noise: where images are convolved with a Gaussian kernel k, and noise is added.
def add_convolve_noise(image, sigma=1.0, mean=0, sigma_noise=25):
    """
    Add Convolve-Noise to an image.
    
    Args:
        image: Input image (numpy array).
        sigma: Standard deviation of the Gaussian kernel.
        mean: Mean of the Gaussian kernel.
        sigma_noise: Standard deviation of the noise to be added.
    
    Returns:
        Noisy image (numpy array).
    """
    # Create a Gaussian kernel
    copy_image = np.copy(image)
    kernel_size = int(6 * sigma) + 1
    gaussian_kernel = cv2.getGaussianKernel(kernel_size, sigma)
    
    # Convolve the image with the Gaussian kernel
    smoothed_image = cv2.filter2D(copy_image, -1, gaussian_kernel)
    
    # Add Gaussian noise to the convolved image
    h, w, c = smoothed_image.shape
    gaussian_noise = np.random.normal(mean, sigma_noise, (h, w, c))
    noisy_image = np.clip(smoothed_image + gaussian_noise, 0, 255).astype(np.uint8)
    
    return noisy_image,image


# TODO: they are the same 
# Keep-Patch: where pixels outside of a randomly chosen k × k patch are set to zero
def add_keep_patch_noise(image, height_patch_size=32,width_patch_size=32):
    """
    Add Keep-Patch noise to an image.
    
    Args:
        image: Input image (numpy array).
        patch_size: Size of the square patch to keep.
    
    Returns:
        Noisy image (numpy array).
    """
    copy_image = np.copy(image)
    h, w, c = image.shape
    
    # Randomly choose the top-left corner of the patch
    top_left_x = np.random.randint(0, w - width_patch_size)
    top_left_y = np.random.randint(0, h - height_patch_size)
    
    # Create a black image of the same size as the original image
    noisy_image = np.zeros_like(copy_image)
    
    # Place the selected patch into the black image
    noisy_image[top_left_y:top_left_y+height_patch_size, top_left_x:top_left_x+width_patch_size] = image[top_left_y:top_left_y+height_patch_size, top_left_x:top_left_x+width_patch_size]
    
    return noisy_image,image
#Extract-Patch: where pixels within a randomly chosen k × k are extracted.
def add_extract_patch_noise(image, height_patch_size=32,width_patch_size=32):
    """
    Add Extract-Patch noise to an image.
    
    Args:
        image: Input image (numpy array).
        patch_size: Size of the square patch to extract.
    
    Returns:
        Noisy image (numpy array).
    """
    copy_image = np.copy(image)
    h, w, c = image.shape
    
    # Randomly choose the top-left corner of the patch
    top_left_x = np.random.randint(0, w - width_patch_size)
    top_left_y = np.random.randint(0, h - height_patch_size)
    
    # Extract the pixels within the patch
    patch = copy_image[top_left_y:top_left_y+height_patch_size, top_left_x:top_left_x+width_patch_size]
    
    # Create a black image of the same size as the original image
    noisy_image = np.zeros_like(copy_image)
    
    # Place the extracted patch into the black image
    noisy_image[top_left_y:top_left_y+height_patch_size, top_left_x:top_left_x+width_patch_size] = patch
    
    return noisy_image,image

# Pad-Rotate-Project: where the padded image is rotated at a random angle about the center
def add_pad_rotate_project_noise(image, max_rotation=30):
    """
    Add Pad-Rotate-Project noise to an image.
    
    Args:
        image: Input image (numpy array).
        max_rotation: Maximum rotation angle in degrees.
    
    Returns:
        Noisy image (numpy array).
    """
    # Get image dimensions
    copy_image = np.copy(image)
    h, w = copy_image.shape[:2]
    
    # Pad the image to ensure no cropping occurs during rotation
    padding = max(h, w) // 2
    padded_image = cv2.copyMakeBorder(copy_image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    # Randomly choose rotation angle
    angle = np.random.uniform(-max_rotation, max_rotation)
    
    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((padded_image.shape[1]//2, padded_image.shape[0]//2), angle, 1)
    
    # Perform rotation
    rotated_image = cv2.warpAffine(padded_image, rotation_matrix, (padded_image.shape[1], padded_image.shape[0]))
    
    # Project rotated image back onto original size
    projected_image = rotated_image[padding:h+padding, padding:w+padding]
    
    return projected_image,image

# Gaussian-Projection: where the image is projected onto a random Gaussian matrix
def add_gaussian_projection_noise(image, sigma=0.1):
    """
    Add Gaussian-Projection noise to an image.
    
    Args:
        image: Input image (numpy array).
        sigma: Standard deviation of the Gaussian vector.
    
    Returns:
        Noisy image (numpy array).
    """
    copy_image = np.copy(image)
    # Generate random Gaussian vectors for each color channel
    noise = np.random.normal(scale=sigma, size=copy_image.shape)
    
    # Add noise to the image
    noisy_image = copy_image + noise
    
    # Clip the pixel values to ensure they are within the valid range [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255)
    
    # Convert to the appropriate data type (uint8)
    noisy_image = noisy_image.astype(np.uint8)
    
    return noisy_image,image

# line strip noise
def add_line_strip_noise(image, strip_width=5, intensity=0.5):
    """
    Add Line-Strip noise to an image.
    
    Args:
        image: Input image (numpy array).
        strip_width: Width of the line strip.
        intensity: Intensity of the noise.
    
    Returns:
        Noisy image (numpy array).
    """
    copy_image = np.copy(image)
    h, w, c = copy_image.shape
    
    # Randomly choose the orientation of the strip
    rand= np.random.rand()
    if rand < 0.5:
        # Horizontal strip
        top_left_y = np.random.randint(0, h - strip_width)
        copy_image[top_left_y:top_left_y+strip_width] = intensity
    else:
        # Vertical strip
        top_left_x = np.random.randint(0, w - strip_width)
        copy_image[:, top_left_x:top_left_x+strip_width] = intensity
    
    return copy_image,image
if __name__ == "__main__":
   
    # Load image
    image = plt.imread("Logo.jpg")
    # noisy_image,image = add_block_pixel_noise(image, probability=0.05)
    # noisy_image,image = add_convolve_noise(image, sigma=1.5, sigma_noise=25) 
    # noisy_image,image = add_keep_patch_noise(image, height_patch_size=image.shape[0]-25,width_patch_size=image.shape[1]-100 ) #keep patch noise is the size we want to keep
    ########### noisy_image,image = add_extract_patch_noise(image, height_patch_size=512,width_patch_size=25)  # Adjust patch size as needed
    # noisy_image,image = add_pad_rotate_project_noise(image, max_rotation=2) 
    # noisy_image,image = add_gaussian_projection_noise(image, sigma=0.1)
    noisy_image,image = add_line_strip_noise(image, strip_width=5, intensity=0.5)
    # Display images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(noisy_image, cmap="gray")
    plt.title("Noisy Image")
    plt.axis("off")
    plt.show()