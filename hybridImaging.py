
from PIL import Image
import numpy
import scipy.signal
import matplotlib.pyplot as plt


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


def box(dimension=5):
    """
    Computes Box kernel

    Parameters
    ----------
    dimension : int
        The dimension of the computed Box kernel

    Returns
    -------
    Box kernel of defined dimension
    """
    w = numpy.ones((dimension, dimension), dtype=float)
    res = (1/float(dimension**2)) * w
    return res


# Retrieving images
dog = Image.open('dog.jpg').convert("L")
snoop_dog = Image.open('snoopdogg.jpg').convert("L")

# Making image dimensions consistent
snoop_dog = snoop_dog.resize((271, 361))

# Creating kernels
gaussian_kernel = gaussian(13, 4)
box_kernel = box(13)

# Low-pass filtering images
dog_low_pass = scipy.signal.convolve2d(dog, gaussian_kernel, boundary='symm', mode='same')
snoop_dog_low_pass = scipy.signal.convolve2d(snoop_dog, gaussian_kernel, boundary='symm', mode='same')

# Constructing low and high frequency portions of hybrid image
low_freq = dog_low_pass
high_freq = snoop_dog - snoop_dog_low_pass

# Creating hybrid image
hybrid = low_freq + high_freq

# Displaying result
plt.subplot(1, 3, 1)
plt.imshow(dog, cmap='gray')
plt.title('Far View Component')
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(snoop_dog, cmap='gray')
plt.title('Close View Component')
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(hybrid, cmap='gray')
plt.title('Hybrid Image')
plt.axis('off')
plt.show()

# Printing kernel used
numpy.set_printoptions(precision=3)
print(f"\n\nKernel used:\n ", gaussian_kernel)
