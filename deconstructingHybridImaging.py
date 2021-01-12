
from PIL import Image
import numpy
import scipy.signal
import matplotlib.pyplot as plt


def intensity_scale(raw_img):
    """
    Scale image intensity to [0,1] with mean value of 0.5 for better visualization

    Parameters
    ----------
    raw_img : ndarray
        2D array containing image uint8 gray levels

    Returns
    -------
    Scaled image
    """
    # Scaling intensity from [min, max] to [0, 1]
    v_min, v_max = raw_img.min(), raw_img.max()
    scaled_im = (raw_img * 1.0 - v_min) / (v_max - v_min)

    # Shifting mean to 0.5
    mean_gray = numpy.mean(scaled_im)
    scaled_im = scaled_im - mean_gray + 0.5

    # Clipping to [0, 1]
    scaled_im = numpy.clip(scaled_im, 0, 1)
    return scaled_im


def gaussian(dimension=5, sigma=1):
    """
    Computes Gaussian kernel

    Parameters
    ----------
    dimension : int
        The dimension of the computed Gaussian kernel
    sigma : int
        The standard deviation of the computed Gaussian kernel

    Returns
    -------
    Gaussian kernel of defined dimension and standard deviation
    """
    kx = numpy.linspace(-1*(dimension - 1) / 2.0, (dimension - 1) / 2.0, dimension)
    x2, y2 = numpy.meshgrid(kx, kx)
    w = numpy.exp((-1/2.0) * (numpy.square(x2) + numpy.square(y2)) / numpy.square(sigma))
    res = w / numpy.sum(w)
    return res


# Retrieving hybrid image
hybrid = Image.open('einsteinandwho.png').convert("L")

# Creating kernel
gaussian_kernel = gaussian(35,10)

# Extracting spatial frequency components from hybrid image
low_freq_image = scipy.signal.convolve2d(hybrid, gaussian_kernel, boundary='symm', mode='same')
high_freq_image = hybrid - low_freq_image

# Scaling image intensity for better visualization
low_freq_image = intensity_scale(low_freq_image)
high_freq_image = intensity_scale(high_freq_image)

# Displaying results
plt.subplot(1, 3, 1)
plt.imshow(hybrid, cmap='gray')
plt.title('Hybrid Image')
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(low_freq_image, cmap='gray')
plt.title('Low Frequency Image')
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(high_freq_image, cmap='gray')
plt.title('High Frequency Image')
plt.axis('off')
plt.show()
