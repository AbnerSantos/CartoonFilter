import numpy as np
import cv2
import os

threshold = 150
num_bilateral = 7  
d = 20
sigma_color = 9
sigma_space = 7

# Reads the image
filename = str(input()).rstrip() 

image = cv2.imread(filename)

# Creates a copy of the image
image_cartoon = np.array(image, copy=True)
image_edges = np.array(image, copy=True)

# Repeatedly applies bilateral filtering to each color channel
for _ in range(num_bilateral):
    # Gets a copy of the color channels
    red   = np.clip(image_cartoon[:,:,0], 0, 255)
    green = np.clip(image_cartoon[:,:,1], 0, 255)
    blue  = np.clip(image_cartoon[:,:,2], 0, 255)

    # Applies filtering
    red   = cv2.bilateralFilter(image_cartoon[:,:,0], d, sigma_color, sigma_space)
    green = cv2.bilateralFilter(image_cartoon[:,:,1], d, sigma_color, sigma_space)
    blue  = cv2.bilateralFilter(image_cartoon[:,:,2], d, sigma_color, sigma_space)

    # Copies to each channel of the cartoonized image
    image_cartoon[:,:,0] = red
    image_cartoon[:,:,1] = green
    image_cartoon[:,:,2] = blue

# Displays the cartoonized image
cv2.imshow("cartoon", image_cartoon)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Mask for the edge detection
kernel = np.array([[-1, -1, -1], [-1,  8, -1], [-1, -1, -1]])

# Applies the convolution
image_edges[:,:,0] = cv2.filter2D(image_edges[:,:,0], -1, kernel)
image_edges[:,:,1] = cv2.filter2D(image_edges[:,:,1], -1, kernel)
image_edges[:,:,2] = cv2.filter2D(image_edges[:,:,2], -1, kernel)

# Displays the unfiltered result of the edge detection
cv2.imshow("edges", image_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Applies threshold
for x in range(image_edges.shape[0]):
    for y in range(image_edges.shape[1]):
        if np.sum(image_edges[x][y]) > 150:
            image_edges[x][y] = (255, 255, 255)
        else:
            image_edges[x][y] = (0, 0, 0)

# Displays the edges after the threshold as white outlines
cv2.imshow("edges", image_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

new_filename = os.path.splitext(filename)[0] + "_cartoonized.png"
cv2.imwrite(new_filename, image_cartoon)