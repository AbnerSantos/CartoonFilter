'''
    Name: Abner Eduardo Silveira Santos
    USP number: 10692012
    Name: Gyovana Mayara Moriyama
    USP number: 10734387
    Name: Henrique Matarazo Camillo
    USP number: 10294943
    Course and Ingress Year: BCC 2018
    Year: 2019
    Semester: 1st
    Course Code: SCC0251
    Title: Cartoon Filter
'''

import numpy as np
import cv2
import os

# Threshold, for edge detection
threshold = 150

# Number of times the 
num_bilateral = 7

# Custom bilateral filter parameters
sigma_s = 4
sigma_r = 0.2
kernel_size = 3

# OpenCV bilateral filter parameters
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

'''
    Applies bilateral filter to an image, and returns it    
        Parameters:
            image       =   original image
            kernel_size =   kernel size
            sigma_s     =   spatial sigma
            sigma_r     =   range sigma
'''
def bilateral_filter(image, kernel_size, sigma_s, sigma_r):
    size_x, size_y = image.shape

    offset = kernel_size // 2

    # Calculates the kernels for sigma_s and sigma_r
    kernel_s = gaussian_filter(kernel_size, sigma_s)
    kernel_r = gaussian_filter(kernel_size, sigma_r)

    new_image = np.array(image, copy=True).astype(float)

    # For each pixel in the image
    for x in range(size_x):
        for y in range(size_y):
            # If the submatrix won't go out of edges
            if not(x - offset < 0 or x + offset >= size_x or y - offset < 0 or y + offset >= size_y):
                # Gets the submatrix
                sub_matrix = image[(x-offset):(x+offset+1), (y-offset):(y+offset+1)]

                # Calculates the two parts of the equation
                spatial_gaussian = np.multiply(sub_matrix, kernel_s)
                range_gaussian   = np.multiply(np.absolute(np.subtract(sub_matrix, image[x,y])), kernel_r) 
                
                # Multiplies the tow parts of the equation, getting a partial result, from where we can get the normalization factor
                result = np.multiply(spatial_gaussian, range_gaussian)
                normalization_factor = np.sum(result)

                # Gets the result matrix multiplying the parcial result by the original sub matrix
                result = np.multiply(result, sub_matrix)
                result = np.sum(result)

                # Normalizes the pixel, if the normalization factor isn't zero
                if (normalization_factor != 0):
                    result = np.divide(result, normalization_factor)
                # If it's zero, doesn't change the pixel
                else:
                    result = image[x,y]

                # The final result is the sum of the result matrix
                new_image[x,y] = result

    # Returns the image
    return new_image

# Reads inputs
filename = str(input("Insert the image filename\n")).rstrip() 
image = cv2.imread(filename)

filter_option = int(input("\nChoose the bilateral filter:\n\t1 - OpenCV Bilateral Filter\n\t2 - Custom Bilateral Filter\n"))
if (filter_option > 2 or filter_option < 1):
    print("Invalid option")
    exit(0)

edges_option = int(input("\nDo you want the image with or without edges?:\n\t1 - With edges\n\t2 - Without edges\n"))
if (edges_option != 1 and edges_option != 2):
    print("Invalid option")
    exit(0)

# Downsamples the image
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
    if filter_option == 1:
        red   = cv2.bilateralFilter(image_cartoon[:,:,0], d, sigma_color, sigma_space)
        green = cv2.bilateralFilter(image_cartoon[:,:,1], d, sigma_color, sigma_space)
        blue  = cv2.bilateralFilter(image_cartoon[:,:,2], d, sigma_color, sigma_space)
    elif filter_option == 2:
        red     = bilateral_filter(image_cartoon[:,:,0], kernel_size,sigma_s, sigma_r)
        green   = bilateral_filter(image_cartoon[:,:,1], kernel_size, sigma_s, sigma_r)
        blue    = bilateral_filter(image_cartoon[:,:,2], kernel_size, sigma_s, sigma_r)


    # Copies to each channel of the cartoonized image
    image_cartoon[:,:,0] = red
    image_cartoon[:,:,1] = green
    image_cartoon[:,:,2] = blue

# Displays the cartoonized image
cv2.imshow("Cartoon", image_cartoon)
cv2.waitKey(0)
cv2.destroyAllWindows()

# If the user chose the image without edges, finishes the program writing the image
if edges_option == 2:
    new_filename = os.path.splitext(filename)[0] + "_cartoon.png"
    cv2.imwrite(new_filename, image_cartoon)

    exit(0)
        
# Else, continues to edge detection
image_cartoon = cv2.cvtColor(image_cartoon, cv2.COLOR_BGR2HSV)

# Applies threshold
for x in range(image_cartoon.shape[0]):
    for y in range(image_cartoon.shape[1]):
        if image_cartoon[x, y][1] * 1.1 > 255:
            image_cartoon[x, y][1] = 255
        else:
            image_cartoon[x, y][1] *= 1.1

image_cartoon = cv2.cvtColor(image_cartoon, cv2.COLOR_HSV2BGR)

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

# Displays the edges after morphological operations
cv2.imshow("edges with treatment", image_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

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

# Writes the image
new_filename = os.path.splitext(filename)[0] + "_cartoon_outline.png"
cv2.imwrite(new_filename, image_cartoon)