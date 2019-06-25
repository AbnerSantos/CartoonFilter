import numpy as np
import cv2
import os

threshold = 150
num_bilateral = 7  
d = 20
sigma_color = 9
sigma_space = 7

'''
#   Gaussian Filter
#   Degradation function used for the image deblurring
#       k: Filter size
#       sigma: Equation parameter/intensity
#       return: Degradation filter
'''
def gaussian_filter (k=3, sigma=1.0):
    arx = np.arange((-k // 2) + 1.0 , (k // 2) + 1.0)
    x , y = np.meshgrid(arx , arx)
    filt = np.exp(-(1/2)*(np.square(x) + np.square(y)) / np.square(sigma))
    return filt / np.sum(filt)

'''
    Generic morphological function
    Used for both dilation and erosion
        kernel: Mask
        base_outline: Raw edges
        func: Function pointer (Max for dilation, Min for erosion)
'''
def morph (kernel, base_outline, func):
    new_outline = np.zeros(base_outline.shape)
    k = kernel.shape[0]
    offset = (k-1)//2
    n = base_outline.shape[0]
    m = base_outline.shape[1]

    for x in range(n):
        for y in range(m):
            if offset <= x < (n-offset) and offset <= y < (m-offset):
                new_outline[x, y] = func(cv2.filter2D(base_outline[(x-offset):(x+offset+1), (y-offset):(y+offset+1)], -1, kernel))
            else:
                new_outline[x, y] = base_outline[x, y]

    return new_outline

def dilation (kernel, base_outline):
    func = np.max
    return morph(kernel, base_outline, func)

def erosion (kernel, base_outline):
    func = np.min
    return morph(kernel, base_outline, func)

# Reads the image
filename = str(input()).rstrip() 

image = cv2.imread(filename)

while image.shape[0] > 512  and image.shape[1] > 512:
    image = cv2.pyrDown(image)

# Creates a copy of the image
image_cartoon = np.array(image, copy=True)

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

image_cartoon = cv2.cvtColor(image_cartoon, cv2.COLOR_BGR2HSV)

# Applies threshold
for x in range(image_cartoon.shape[0]):
    for y in range(image_cartoon.shape[1]):
        if image_cartoon[x, y][1] * 1.1 > 255:
            image_cartoon[x, y][1] = 255
        else:
            image_cartoon[x, y][1] *= 1.1

image_cartoon = cv2.cvtColor(image_cartoon, cv2.COLOR_HSV2BGR)

# Displays the cartoonized image
cv2.imshow("cartoon", image_cartoon)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Creates a gray copy of the cartoon image
image_edges = np.array(image_cartoon, copy=True)
image_edges = cv2.cvtColor(image_edges, cv2.COLOR_BGR2GRAY)

# Mask for the edge detection
kernel = np.array([[-1, -1, -1], [-1,  8, -1], [-1, -1, -1]])
image_edges = cv2.filter2D(image_edges, -1, gaussian_filter())

# Applies the convolution
image_edges = cv2.filter2D(image_edges, -1, kernel)

# Displays the unfiltered result of the edge detection
cv2.imshow("edges", image_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Applies threshold
for x in range(image_edges.shape[0]):
    for y in range(image_edges.shape[1]):
        if image_edges[x][y] > 5:
            image_edges[x, y] = 255
        else:
            image_edges[x, y] = 0

# Displays the edges after the threshold
cv2.imshow("edges with threshold", image_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
image_edges = erosion(kernel, image_edges)
# image_edges = dilation(kernel, image_edges)

# Displays the edges after morphological operations
cv2.imshow("edges with treatment", image_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

new_filename = os.path.splitext(filename)[0] + "_cartoon.png"
cv2.imwrite(new_filename, image_cartoon)

image_levels = cv2.cvtColor(image_cartoon, cv2.COLOR_BGR2GRAY)
image_cartoon = cv2.cvtColor(image_cartoon, cv2.COLOR_BGR2HSV)

# Applies threshold
for x in range(image_edges.shape[0]):
    for y in range(image_edges.shape[1]):
        if image_edges[x, y] > 0:
            image_cartoon[x, y][1] *= 0.9

            if image_cartoon[x, y][2] > 200:
                image_cartoon[x, y][2] *= 0.8
            else: 
                if image_cartoon[x, y][2] * 1.2 > 255:
                    image_cartoon[x, y][2] = 255
                else:
                    image_cartoon[x, y][2] *= 1.2

image_cartoon = cv2.cvtColor(image_cartoon, cv2.COLOR_HSV2BGR)

# Displays the edges after the threshold
cv2.imshow("Image with edges", image_cartoon)
cv2.waitKey(0)
cv2.destroyAllWindows()

new_filename = os.path.splitext(filename)[0] + "_cartoon_outline.png"
cv2.imwrite(new_filename, image_cartoon)