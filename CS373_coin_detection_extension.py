# Built in packages
import math
import sys

# Matplotlib will need to be installed if it isn't already. This is the only package allowed for this base part of the 
# assignment.
from matplotlib import pyplot
from matplotlib.patches import Rectangle

# import our basic, light-weight png reader library
import imageIO.png

# Define constant and global variables
TEST_MODE = False    # Please, DO NOT change this variable!

def readRGBImageToSeparatePixelArrays(input_filename):
    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)

# a useful shortcut method to create a list of lists based array representation for an image, initialized with a value
def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):
    new_pixel_array = []
    for _ in range(image_height):
        new_row = []
        for _ in range(image_width):
            new_row.append(initValue)
        new_pixel_array.append(new_row)

    return new_pixel_array


###########################################
### You can add your own functions here ###
###########################################

### 1. GREYSCALE AND NORMALIZE ###
def convertToGreyscaleAndNormalize(pixel_array_r, pixel_array_g, pixel_array_b, image_width, image_height):
    greyscale_array = createInitializedGreyscalePixelArray(image_width, image_height)
    
    # Greyscale #
    for h in range(image_height):   # Iterate over each pixel
        for w in range(image_width):
            r = pixel_array_r[h][w]
            g = pixel_array_g[h][w]
            b = pixel_array_b[h][w]
            grey = 0.3*r + 0.6*g + 0.1*b
            greyscale_array[h][w] = round(grey)

    # Cumulative histogram #
    histo = [0]*256
    for h in range(image_width):
        for w in range(image_height):
            histo[greyscale_array[w][h]] += 1
            
    for i in range(1, len(histo)):
        histo[i] += histo[i-1]
    
    # Normalize #
    greyscale_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)
    
    # Find the nth index of the first value in cumulative where its value is at 5%
    total = histo[-1]
    fmin = 0
    for i in range(256):
        if histo[i] >= total*0.05:
            fmin = i
            break
    
    # Find the nth index of the first value in cumulative where its value is at 95%
    fmax = 0
    for i in range(255, -1, -1):
        if histo[i] <= total*0.95:
            fmax = i
            break

    for h in range(image_height):
        for w in range(image_width):
            if greyscale_array[h][w] < fmin:
                greyscale_pixel_array[h][w] = 0
            elif greyscale_array[h][w] > 190:
                greyscale_pixel_array[h][w] = 255
            else:            
                greyscale_pixel_array[h][w] = (greyscale_array[h][w] - fmin) * (255.0 / (fmax - fmin))
                
    return greyscale_pixel_array



### 2. EDGE DETECTION ###
def LaplacianFilter(pixel_array, image_width, image_height):
    edge_array = createInitializedGreyscalePixelArray(image_width, image_height)
    kernel = [
        [1.0, 1.0, 1.0],
        [1.0, -8.0, 1.0],
        [1.0, 1.0, 1.0]
    ]
    for h in range(1, image_height - 1):
        for w in range(1, image_width - 1):
            result = 0

            # Applying kernel to neighbouring pixels
            for i in range(-1, 2):
                for j in range(-1, 2):
                    result += kernel[i + 1][j + 1] * pixel_array[h + i][w + j]
            edge_array[h][w] = min(max(result, 0), 255)     # Result must be in range of 0 to 255
    
    return edge_array



### 3. BLURRING ###
def MediumFilter(pixel_array, image_width, image_height):
    array = createInitializedGreyscalePixelArray(image_width, image_height)
    
    for h in range(image_height):
        for w in range(image_width):
            kernel = []

            # Iterate over 3x5 neighbourhood of current pixel
            for i in range(-1, 2):
                for j in range(-2, 3):
                    hi = h + i
                    wj = w + j
                    if 0 <= hi < image_height and 0 <= wj < image_width:    # Check if neighbour within image boundary
                        kernel.append(pixel_array[hi][wj])                  # Append the pixel value to the kernel list if in bounds
                    else:
                        kernel.append(0)                                    # Append 0 if out of bounds
                        
            array[h][w] = int(sorted(kernel)[len(kernel) // 2])             # Sort kernel values. Assign median value to output pixel
    return array


def GaussianFilter(pixel_array, image_width, image_height):
    array = createInitializedGreyscalePixelArray(image_width, image_height)
    # 5x5 Gaussian Kernel
    kernel = [
        [1, 4, 7, 4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1, 4, 7, 4, 1]
    ]
    
    centre = len(kernel) // 2   # Centre position of kernel
    for h in range(centre, image_height - centre):
        for w in range(centre, image_width - centre):
            result = 0.0

            # Apply the Gaussian kernel to current and neighbouring pixels
            for i in range(len(kernel)):
                for j in range(len(kernel)):
                    result += kernel[i][j] * pixel_array[h + i - centre][w + j - centre]
            
            # Normalize result. Sum of kernel values = 273
            array[h][w] = result / 273.0
    
    # Convert each pixel in the output array to float
    return [[float(pixel) for pixel in row] for row in array]



### Thresholding ###
def adaptiveThreshold(pixel_array, image_width, image_height):
    histogram = [0] * 256

    # The frequency of each pixel intensity
    for h in range(image_height):
        for w in range(image_width):
            num = int(pixel_array[h][w])
            histogram[num] += 1
    
    q_values = list(range(256))     # List of intensity values
    theta_0 = sum(q * histogram[q] for q in q_values) / sum(histogram)      # Initial threshold
    
    theta_j = theta_0
    theta_j1 = -1
    
    while theta_j != theta_j1:      # Iterate until both values are the same
        theta_j1 = theta_j          # Updating variable
        
        px_obj = sum(histogram[q] for q in range(int(theta_j) + 1))         # No. of pixels belong to object
        px_bg = sum(histogram[q] for q in range(int(theta_j) + 1, 256))     # No. of pixels belong to background
        
        if px_obj == 0 or px_bg == 0:      # Prevent zero division
            break
        
        av_obj = sum(q * histogram[q] for q in range(int(theta_j) + 1)) / px_obj        # Average intensity of object
        av_bg = sum(q * histogram[q] for q in range(int(theta_j) + 1, 256)) / px_bg     # Average intensity of background
        
        theta_j = 0.5 * (av_obj + av_bg)    # Midpoint between the two means
    
    threshold = theta_j
 
    threshold_array = []
    for row in pixel_array:     # Apply threshold to each input
        thresholded_row = [0 if pixel < threshold else 255 for pixel in row]    # If lower than threshold, set to 0. Otherwise, 255
        threshold_array.append(thresholded_row)
    
    return threshold_array


### Dilation and Errosion ###
def Dilation(pixel_array, image_width, image_height):
    dilated_array = createInitializedGreyscalePixelArray(image_width, image_height)
    kernel = [              # Changed kernel to square instead of circle
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1]
    ]
    
    # Iterate over each pixel, excluding the borders; BoundaryZeroPadding
    for h in range(2, image_height - 2):
        for w in range(2, image_width - 2):
            # Max value in neighbour
            max_val = 0

            # Apply the kernel to the neighbourhood of the current pixel
            for i in range(-2, 3):
                for j in range(-2, 3):
                    if kernel[i + 2][j + 2] == 1:     # Only check pixels where the kernel has a value of 1
                        max_val = max(max_val, pixel_array[h + i][w + j])     # Update max_val to be the maximum value in the neighbourhood
            
            # Set current pixel to max value found
            dilated_array[h][w] = max_val
    return dilated_array

def Erosion(pixel_array, image_width, image_height):
    kernel = [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0]
    ]
    eroded_array = createInitializedGreyscalePixelArray(image_width, image_height)
    for h in range(2, image_height - 2):
        for w in range(2, image_width - 2):
            min_val = 255
            for kh in range(-2, 3):
                for kw in range(-2, 3):
                    if kernel[kh + 2][kw + 2] == 1:
                        min_val = min(min_val, pixel_array[h + kh][w + kw])
            eroded_array[h][w] = min_val
    return eroded_array

def MorphologicalGradient(pixel_array, image_width, image_height):
    dilate = Dilation(pixel_array, image_width, image_height)
    erode = Erosion(pixel_array, image_width, image_height)
    array = createInitializedGreyscalePixelArray(image_width, image_height)
    
    # Across each pixel: Dilation - Erosion
    for h in range(image_height):
        for w in range(image_width):
            array[h][w] = abs(dilate[h][w]) - abs(erode[h][w])
    
    return array

### Connected Component ###
class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0, item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)



def ConnectedComponent(pixel_array, image_width, image_height):
    result_image = createInitializedGreyscalePixelArray(image_width, image_height)
    labels = {}         # Store the sizes of the connected components
    current_label = 1   # Initialize the current label to 1

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]     # All possible directions for 4-connectivity
    
    for h in range(image_height):
        for w in range(image_width):
            # If the pixel that's part of the component is not labelled
            if pixel_array[h][w] > 0 and result_image[h][w] == 0:
                # Initialise queue for BFS
                q = Queue()
                q.enqueue((h, w))

                # Size of current component
                size = 0
                
                # Perform BFS to label all connected pixels
                while not q.isEmpty():
                    x, y = q.dequeue()
                    if result_image[x][y] == 0:
                        result_image[x][y] = current_label  # Label current pixel with current label
                        size += 1
                        
                        # Check all 4-connected neighbours
                        for directionx, directiony in directions:
                            neighbourx = x + directionx
                            neighboury = y + directiony
                            
                            if 0 <= neighbourx < image_height and 0 <= neighboury < image_width:    # Check if neighbour is within image bounds
                                if pixel_array[neighbourx][neighboury] > 0 and result_image[neighbourx][neighboury] == 0: # Check if neighbour is part of the component and not labeled
                                    q.enqueue((neighbourx, neighboury))
                
                # Store size of current component
                labels[current_label] = size
                current_label += 1
    
    # Store the coordinates of each component
    components = {} 
    for h in range(image_height):
        for w in range(image_width):
            label = result_image[h][w]
            if label > 0:
                if label not in components:
                    components[label] = []
                components[label].append((h, w))

    # Store the bounding boxes of each component
    bounding_boxes = []
    for label, pixels in components.items():
        min_x = min(p[1] for p in pixels)
        min_y = min(p[0] for p in pixels)
        max_x = max(p[1] for p in pixels)
        max_y = max(p[0] for p in pixels)
        bounding_boxes.append((min_x, min_y, max_x, max_y))
    
    return bounding_boxes



# This is our code skeleton that performs the coin detection.
def main(input_path, output_path):
    # This is the default input image, you may change the 'image_name' variable to test other images.
    image_name = 'hard_case_1'
    input_filename = f'./Images/hard/{image_name}.png'
    # image_name = 'easy_case_6'
    # input_filename = f'./Images/easy/{image_name}.png'
    if TEST_MODE:
        input_filename = input_path

    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(input_filename)
    
    ###################################
    ### STUDENT IMPLEMENTATION Here ###
    ###################################
    array = convertToGreyscaleAndNormalize(px_array_r, px_array_g, px_array_b, image_width, image_height)
    array = GaussianFilter(array, image_width, image_height)
    array = GaussianFilter(array, image_width, image_height)
    array = GaussianFilter(array, image_width, image_height)
    array = LaplacianFilter(array, image_width, image_height)
    array = MediumFilter(array, image_width, image_height)
    array = adaptiveThreshold(array, image_width, image_height)
    array = Dilation(array, image_width, image_height)
    array = Dilation(array, image_width, image_height)
    array = MorphologicalGradient(array, image_width, image_height)
    bounding_box_list = ConnectedComponent(array, image_width, image_height)

    ############################################
    ### Bounding box coordinates information ###
    ### bounding_box[0] = min x
    ### bounding_box[1] = min y
    ### bounding_box[2] = max x
    ### bounding_box[3] = max y
    ############################################
    
    # bounding_box_list = [[150, 140, 200, 190]]  # This is a dummy bounding box list, please comment it out when testing your own code.
    # px_array = px_array_r
    px_array = pyplot.imread(input_filename)

    fig, axs = pyplot.subplots(1, 1)
    axs.imshow(px_array, aspect='equal')
    
    # Loop through all bounding boxes
    for bounding_box in bounding_box_list:
        bbox_min_x = bounding_box[0]
        bbox_min_y = bounding_box[1]
        bbox_max_x = bounding_box[2]
        bbox_max_y = bounding_box[3]
        
        bbox_xy = (bbox_min_x, bbox_min_y)
        bbox_width = bbox_max_x - bbox_min_x
        bbox_height = bbox_max_y - bbox_min_y
        if 320 > bbox_width > 200 and 320 > bbox_height > 200:  # Eliminates unnecessary boundary boxes
            rect = Rectangle(bbox_xy, bbox_width, bbox_height, linewidth=2, edgecolor='r', facecolor='none')
            axs.add_patch(rect)
            if 238 <= bbox_height <= 250 and 240 <= bbox_width <= 254:      # Size range of coins
                pyplot.text(bbox_min_x+4, bbox_max_y-4, "$1")
            elif 272 <= bbox_height <= 290 and 273 <= bbox_width <= 290:
                pyplot.text(bbox_min_x+4, bbox_max_y-4, "$2")
            elif 251 <= bbox_height <= 271 and 255 <= bbox_width <= 272:
                pyplot.text(bbox_min_x+4, bbox_max_y-4, "50c")
            elif 210 <= bbox_height <= 223 and 205 <= bbox_width <= 226:
                pyplot.text(bbox_min_x+4, bbox_max_y-4, "10c")
            elif 224 <= bbox_height <= 237 and 227 <= bbox_width <= 240:
                pyplot.text(bbox_min_x+4, bbox_max_y-4, "20c")
        
    pyplot.axis('off')
    pyplot.tight_layout()
    default_output_path = f'./output_images/{image_name}_with_bbox.png'
    if not TEST_MODE:
        # Saving output image to the above directory
        pyplot.savefig(default_output_path, bbox_inches='tight', pad_inches=0)
        
        # Show image with bounding box on the screen
        pyplot.imshow(px_array, cmap='gray', aspect='equal')
        pyplot.show()
    else:
        # Please, DO NOT change this code block!
        pyplot.savefig(output_path, bbox_inches='tight', pad_inches=0)

if __name__ == "__main__":
    num_of_args = len(sys.argv) - 1
    
    input_path = None
    output_path = None
    if num_of_args > 0:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        TEST_MODE = True
    
    main(input_path, output_path)