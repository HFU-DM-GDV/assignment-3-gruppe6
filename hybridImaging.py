import numpy as np
import cv2 as cv

# Load the left and right images
img_highpass = cv.imread('./images/angry_man.png', cv.IMREAD_GRAYSCALE)
img_lowpass = cv.imread('./images/woman.png', cv.IMREAD_GRAYSCALE)

def get_frequencies(image):
    """
    Compute spectral image with a DFT.
    """
    # Convert image to floats and do dft saving as complex output
    dft = cv.dft(np.float32(image), flags=cv.DFT_COMPLEX_OUTPUT)

    # Apply shift of origin from upper left corner to center of image
    dft_shift = np.fft.fftshift(dft)

    # Extract magnitude and phase images
    mag, phase = cv.cartToPolar(dft_shift[:, :, 0], dft_shift[:, :, 1])

    # Get spectrum for viewing only
    spec = (1 / 20) * np.log(mag)

    # Return the resulting image (as well as the magnitude and
    # Phase for the inverse)
    return spec, mag, phase


def create_from_spectrum(mag, phase):
    # Convert magnitude and phase into cartesian real and imaginary components
    real, imag = cv.polarToCart(mag, phase)

    # Combine cartesian components into one complex image
    back = cv.merge([real, imag])

    # Shift origin from center to upper left corner
    back_ishift = np.fft.ifftshift(back)

    # Do idft saving as complex output
    img_back = cv.idft(back_ishift)

    # Combine complex components into original image again
    img_back = cv.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # Re-normalize to 8-bits
    min, max = np.amin(img_back, (0, 1)), np.amax(img_back, (0, 1))
    print(min, max)
    img_back = cv.normalize(img_back, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    return img_back

# Compute the magnitude and phase of the DFT of the images
result_highpass, mag_h, phase_h = get_frequencies(img_highpass)

row, col = result_highpass.shape
crow, ccol = int(row / 2), int(col / 2)
result_highpass[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0

reconverted_highpass = create_from_spectrum(mag_h, phase_h)

result_lowpass, mag_l, phase_l = get_frequencies(img_lowpass)

# Tiefpassfilter anwenden
rows, cols = result_lowpass.shape
crow, ccol = rows // 2, cols // 2  # Zentrum des Bildes
d = 30  # Radius des Tiefpassfilters
mask = np.zeros((rows, cols), np.uint8)
mask[crow - d:crow + d, ccol - d:ccol + d] = 1
mag_l_filtered = mag_l * mask

reconverted_lowpass = create_from_spectrum(mag_l_filtered, phase_l)

# overlap images


overlap = reconverted_lowpass + reconverted_highpass

# Display the image
cv.imshow('disparity', reconverted_lowpass)
cv.imshow('woman', reconverted_highpass)
cv.imshow('overlap', overlap)
if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()