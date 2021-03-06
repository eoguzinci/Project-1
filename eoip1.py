#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
# %matplotlib inline

#reading in an image
image = mpimg.imread('test_images/whiteCarLaneSwitch.jpg')
#printing out some stats and plotting
print('This image is:', type(image), 'with dimesions:', image.shape)
#plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')

import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color, thickness):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    return line_img

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)
    
#def mask_pixels(x,y, vertices):
#    fit_top = np.polyfit((vertices[1][0], vertices[2][0]), (vertices[1][1], vertices[2][1]), 1)
#    fit_left = np.polyfit((vertices[0][0], vertices[1][0]), (vertices[0][1], vertices[1][1]), 1)
#    fit_right = np.polyfit((vertices[3][0], vertices[2][0]), (vertices[3][1], vertices[2][1]), 1)
#    fit_bottom = np.polyfit((vertices[0][0], vertices[3][0]), (vertices[0][1], vertices[3][1]), 1)
#    
#    XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
#    region_thresholds = (YY > (XX*fit_top[0] + fit_top[1])) & \
#                        (YY > (XX*fit_left[0] + fit_left[1])) & \
#                        (YY > (XX*fit_right[0] + fit_right[1])) & \
#                        (YY < (XX*fit_bottom[0] + fit_bottom[1]))    

import os
os.listdir("test_images/")



# Test on images
# TO DO
def process_image(image):
    gray = grayscale(image)
    blur_gray = gaussian_blur(gray,5)
    edges = canny(blur_gray, 50, 150)
    
    region_select = np.copy(edges)
    
    ysize = edges.shape[0]
    xsize = edges.shape[1]
    margin = 50.0
    
    left_bottom = [0, ysize]
    right_bottom = [xsize, ysize]
    right_top = [xsize/2+margin,320]
    left_top = [xsize/2-margin,320]
    
    vertices = np.array([[left_bottom,left_top, right_top, right_bottom]], dtype=np.int32)
    
    region = region_of_interest(image,vertices)
    fig3 = plt.figure()
    plt.imshow(region)
    
    
    region_select = region_of_interest(region_select,vertices)
    fig4 = plt.figure()
    plt.imshow(region_select, cmap='Greys_r')
    plt.savefig('dashed_line.png')
    
    lines = hough_lines(region_select, 2, 1*np.pi/180, 20, 2, 2)
    line_image = draw_lines(image, lines,[255, 0, 0],2)
    
    fig5 = plt.figure()
    plt.imshow(line_image)
    
    combo = weighted_img(line_image, image)
    
    fig6 = plt.figure()
    plt.imshow(combo)
    plt.savefig('image_test.png')
    
    lines = np.squeeze(lines)
    slope = (lines[:,3]-lines[:,1]) / (lines[:,2] - lines[:,0])
    # line_size = np.sqrt((lines[:,2] - lines[:,0])**2 + (lines[:,3]-lines[:,1])**2)
    
    # Eliminate horizontal lines
    
    tol = 0.2 # tolerance to determine and eliminate horizontal lines
    index = np.abs(slope)>tol
    lines = lines[index]
    slope = (lines[:,3]-lines[:,1]) / (lines[:,2] - lines[:,0])
    line_size = np.sqrt((lines[:,2] - lines[:,0])**2 + (lines[:,3]-lines[:,1])**2)
    
    #sahpe = lines.shape
    #print(sahpe[0], sahpe[1])
    #linnes = lines.reshape(sahpe[0],1,sahpe[1])
    #line_image = draw_lines(image, linnes)
    #
    #fig7 = plt.figure()
    #plt.imshow(line_image)
    
    # Separate Right and Left lanes
    lines_right = lines[slope>0]
    lines_left = lines[slope<0]
    slope_right = slope[slope>0]
    slope_left = slope[slope<0]
    line_size_R = line_size[slope>0]
    line_size_L = line_size[slope<0]
    
    # Average the slopes & longest lines
    index_lensort_right = np.argsort(line_size_R) #be aware that this sorts ascending! 
    index_lensort_left = np.argsort(line_size_L)
    inlen_slope_R = slope_right[index_lensort_right]
    inlen_slope_L = slope_left[index_lensort_left]
    
    threshold_length = 6
    average_slope_R = inlen_slope_R[-threshold_length::].mean()
    average_slope_L = inlen_slope_L[-threshold_length::].mean()
    
    
    longest = 6
    longest_lines_index_L = index_lensort_left[-longest:]
    longest_lines_index_R = index_lensort_right[-longest:]
    
    longest_lines_R = lines_right[longest_lines_index_R]
    longest_lines_L = lines_left[longest_lines_index_L]
    len_long_R = line_size_R[longest_lines_index_R]
    len_long_L = line_size_L[longest_lines_index_L]
    
    # midpooints in L and R
    midpoint_R = np.zeros(2)
    midpoint_L = np.zeros(2)
    midpoint_R[0] = np.sum((longest_lines_R[:,0]+longest_lines_R[:,2])*len_long_R[:]/2)/np.sum(len_long_R[:])
    midpoint_R[1] = np.sum((longest_lines_R[:,1]+longest_lines_R[:,3])*len_long_R[:]/2)/np.sum(len_long_R[:])
    midpoint_L[0] = np.sum((longest_lines_L[:,0]+longest_lines_L[:,2])*len_long_L[:]/2)/np.sum(len_long_L[:])
    midpoint_L[1] = np.sum((longest_lines_L[:,1]+longest_lines_L[:,3])*len_long_L[:]/2)/np.sum(len_long_L[:])
    
    # Line functions for L and R
    bR = midpoint_R[1]-average_slope_R*midpoint_R[0]
    bL = midpoint_L[1]-average_slope_L*midpoint_L[0] 
    
    xR = np.arange(xsize)
    yR = average_slope_R*xR+bR
    # lineL = np.concatenate((xL,yL), axis=1)
    
    xL = np.arange(xsize)
    yL = average_slope_L*xL+bL
    # lineL = np.concatenate((xL,yL), axis=1)
    
    yL_mod2 = yL[yL>320]
    yL_mod = yL_mod2[yL_mod2<ysize]
    yL_max = int(np.max(yL_mod))
    xL_max = int((yL_max-bL)/average_slope_L)
    yL_min = int(np.min(yL_mod))
    xL_min = int((yL_min-bL)/average_slope_L)
    
    yR_mod2 = yR[yR>320]
    yR_mod = yR_mod2[yR_mod2<ysize]
    yR_max = int(np.max(yR_mod))
    xR_max = int((yR_max-bR)/average_slope_R)
    yR_min = int(np.min(yR_mod))
    xR_min = int((yR_min-bR)/average_slope_R)
    
    lines_final = np.array([[xL_min, yL_min, xL_max, yL_max], 
                      [xR_min, yR_min, xR_max, yR_max]])
    lines = lines_final.reshape(lines_final.shape[0],1,lines_final.shape[1])
            
    line_img = draw_lines(image, lines, [255,0,0], 10)
    final_img = weighted_img(image,line_img)
    
    fig7 = plt.figure()
    plt.imshow(final_img)
    
    return final_img
    
from moviepy.editor import VideoFileClip
from IPython.display import HTML

process_image(image)

white_output = 'white.mp4'
clip1 = VideoFileClip("solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))


#longest_lines = np.concatenate((longest_lines_R, longest_lines_L), axis=0)
#sahpe = longest_lines.shape
#linnes = longest_lines.reshape(sahpe[0],1,sahpe[1])
#line_image = draw_lines(image, linnes)
#
#fig7 = plt.figure()
#plt.imshow(line_image)
#plt.savefig('longest_lines.png')




## Test on videos
## TO DO
#
## Import everything needed to edit/save/watch video clips
#from moviepy.editor import VideoFileClip
#from IPython.display import HTML
#
#def process_image(image):
#    # NOTE: The output you return should be a color image (3 channel) for processing video below
#    # TODO: put your pipeline here,
#    # you should return the final output (image with lines are drawn on lanes)
#    
#    return result