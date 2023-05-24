import numpy as np
import cv2 as cv

# Load the left and right images
img_highpass = cv.imread('./images/winnie.png', cv.IMREAD_GRAYSCALE)
img_lowpass = cv.imread('./images/xi.png', cv.IMREAD_GRAYSCALE)

# Resize the images to be of equal size
img_highpass = cv.resize(img_highpass, (img_lowpass.shape[1], img_lowpass.shape[0]))

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

def click_src(event, x, y, flags, param):
    # Grab references to the global variables
    global ref_pt_src

    # If the left mouse button was clicked, add the point to the source array
    if event == cv.EVENT_LBUTTONDOWN:
        pos = len(ref_pt_src)
        if pos == 0:
            ref_pt_src = [(x, y)]
        else:
            ref_pt_src.append((x, y))
        # Draw a circle around the clicked point
        cv.circle(img_lowpass, ref_pt_src[pos], 4, (0, 255, 0), 2)
        cv.imshow("Lowpass", img_lowpass)


def click_dst(event, x, y, flags, param):
    # Grab references to the global variables
    global ref_pt_dst

    # If the left mouse button was clicked, add the point to the destination array
    if event == cv.EVENT_LBUTTONDOWN:
        pos = len(ref_pt_dst)
        if pos == 0:
            ref_pt_dst = [(x, y)]
        else:
            ref_pt_dst.append((x, y))
        # Draw a circle around the clicked point
        cv.circle(img_highpass, ref_pt_dst[pos], 4, (0, 255, 0), 2)
        cv.imshow("Lowpass", dst_transform)

def create_hybrid_image(img1, img2):
    # Compute the magnitude and phase of the DFT for the highpass image
    result_highpass, mag_h, phase_h = get_frequencies(img2)

    # Highpassfilter anwenden
    row, col = result_highpass.shape
    crow, ccol = int(row / 2), int(col / 2)
    result_highpass[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0

    reconverted_highpass = create_from_spectrum(mag_h, phase_h)

    # Compute the magnitude and phase of the DFT for the lowpass image
    result_lowpass, mag_l, phase_l = get_frequencies(img1)

    # Tiefpassfilter anwenden
    rows, cols = result_lowpass.shape
    crow, ccol = rows // 2, cols // 2  # Zentrum des Bildes
    d = 30  # Radius des Tiefpassfilters
    # draw black boxes around middle
    result_lowpass[0:rows, 0:ccol - d] = 0
    result_lowpass[0:rows, ccol + d:cols] = 0
    result_lowpass[0:crow - d, 0:cols] = 0
    result_lowpass[crow + d:rows, 0:cols] = 0
    mag_l_filtered = mag_l * result_lowpass

    reconverted_lowpass = create_from_spectrum(mag_l_filtered, phase_l)

    # overlap images
    alpha = 0.85  # Gewichtung für das Tiefpassbild
    beta = 0.15   # Gewichtung für das Hochpassbild
    hybrid_image = cv.addWeighted(reconverted_lowpass, alpha, reconverted_highpass, beta, 0)
    
    # Display the image
    cv.imshow('hybrid image', hybrid_image)

# Define global arrays for the clicked (reference) points
ref_pt_src = []
ref_pt_dst = []

# Initialize needed variables and windows
rows, cols = img_lowpass.shape
clone_lowpass = img_lowpass.copy()
clone_highpass = img_highpass.copy()
dst_transform = np.zeros(img_lowpass.shape, np.uint8)

cv.namedWindow("Lowpass")
cv.setMouseCallback("Lowpass", click_src)
cv.namedWindow("Highpass")
cv.setMouseCallback("Highpass", click_dst)

# Keep looping until the 'q' key is pressed
computationDone = False
while True:
    # If there are three reference points, then compute the transform and apply the transformation
    if not (computationDone) and (len(ref_pt_src) == 3 and len(ref_pt_dst) == 3):
        T_affine = cv.getAffineTransform(np.float32(ref_pt_src), np.float32(ref_pt_dst))

        print("\nAffine transformation:\n",
              "\n".join(["\t".join(["%03.3f" % cell for cell in row]) for row in T_affine]))

        dst_transform = cv.warpAffine(img_lowpass, T_affine, (cols, rows))
        computationDone = True

    # Display the image and wait for a keypress
    cv.imshow("Lowpass", img_lowpass)
    cv.imshow("Highpass", img_highpass)
    cv.imshow("Lowpass Transformed", dst_transform)

    key = cv.waitKey(10)
    # If the 'r' key is pressed, reset the transformation
    if key == ord("r"):
        dst_transform = np.zeros(img_lowpass.shape, np.uint8)
        img_lowpass = clone_lowpass.copy()
        img_highpass = clone_highpass.copy()
        ref_pt_src = []
        ref_pt_dst = []
        computationDone = False
    elif key == ord("c"):
        img_lowpass = clone_lowpass.copy()
        img_highpass = clone_highpass.copy()
        create_hybrid_image(dst_transform, img_highpass)
    # If the 'q' key is pressed, break from the loop
    elif key == ord("q"):
        break

cv.destroyAllWindows()