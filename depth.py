#ROAD LANE DETECTION

import cv2
import matplotlib.pyplot as plt
import numpy as np

# image_path = "./self-driving-car/project_1_lane_finding_basic/data/test_images/solidWhiteCurve.jpg"#r"/content/road_lane.jpeg"
# image1 = cv2.imread(image_path)
# image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
# plt.imshow(image1)

def grey(image):
  #convert to grayscale
    image = np.asarray(image)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

  #Apply Gaussian Blur --> Reduce noise and smoothen image
def gauss(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

  #outline the strongest gradients in the image --> this is where lines in the image are
def canny(image):
    edges = cv2.Canny(image,50,150)
    return edges

def region(image):
    height, width = image.shape
    #isolate the gradients that correspond to the lane lines
    triangle = np.array([
                       [(0, height), (160, 290), (320, 290), (600, height)]
                       ])
    #create a black image with the same dimensions as original image
    mask = np.zeros_like(image)
    #create a mask (triangle that isolates the region of interest in our image)
    mask = cv2.fillPoly(mask, triangle, 255)
    mask = cv2.bitwise_and(image, mask)
    return mask

def display_lines(image, lines):
    lines_image = np.zeros_like(image)
    #make sure array isn't empty
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            print(f"Lane line coordinates: x1 {x1}, y1 {y1}, x2 {x2}, y2 {y2}")
            #draw lines on a black image
            cv2.line(lines_image, (x1, y1), (x2, y2), (0, 240, 0), 3)
    return lines_image

def average(image, lines):
    left = []
    right = []

    if lines is not None:


        for line in lines:


            # print(line)
            x1, y1, x2, y2 = line.reshape(4)
            #fit line to points, return slope and y-int
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            # print(parameters)
            slope = parameters[0]
            y_int = parameters[1]
            #lines on the right have positive slope, and lines on the left have neg slope
            if slope < 0:
                left.append((slope, y_int))
            else:
                right.append((slope, y_int))

    #takes average among all the columns (column0: slope, column1: y_int)

    right_avg = np.average(right, axis=0)
    left_avg = np.average(left, axis=0)
    #create lines based on averages calculates
    left_line = make_points(image, left_avg)
    right_line = make_points(image, right_avg)
    return np.array([left_line, right_line])

def make_points(image, average):
    # import pdb
    # pdb.set_trace()
    # print(np.array(average).shape)
    slope, y_int = average
    y1 = image.shape[0]
    #how long we want our lines to be --> 3/5 the size of the image
    y2 = int(y1 * (3/5))
    #determine algebraically
    x1 = int((y1 - y_int) // slope)
    x2 = int((y2 - y_int) // slope)
    return np.array([x1, y1, x2, y2])

# from google.colab.patches import cv2_imshow

'''##### DETECTING lane lines in image ######'''

# copy = np.copy(image1)
# edges = cv2.Canny(copy,50,150)
# isolated = region(edges)
# cv2_imshow(edges)
# cv2_imshow(isolated)
# cv2.waitKey(0)


# #DRAWING LINES: (order of params) --> region of interest, bin size (P, theta), min intersections needed, placeholder array,
# lines = cv2.HoughLinesP(isolated, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
# averaged_lines = average(copy, lines)
# black_lines = display_lines(copy, averaged_lines)
# #taking wighted sum of original image and lane lines image
# lanes = cv2.addWeighted(copy, 0.8, black_lines, 1, 1)
# plt.imshow(lanes)
# cv2.waitKey(0)


# Creating a VideoCapture object to read the video
# cap = cv2.VideoCapture('Codes/temp/data/Video/pexels-gentayangan-9265063.mp4')


# Loop until the end of the video
# while (cap.isOpened()):

#     # Capture frame-by-frame
#     ret, frame = cap.read()

#     copy = np.copy(frame)
#     edges = cv2.Canny(copy,50,150)
#     isolated = region(edges)

#     # cv2.waitKey(0)

#     #DRAWING LINES: (order of params) --> region of interest, bin size (P, theta), min intersections needed, placeholder array,
#     lines = cv2.HoughLinesP(isolated, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
#     averaged_lines = average(copy, lines)
#     black_lines = display_lines(copy, averaged_lines)
#     #taking wighted sum of original image and lane lines image
#     lanes = cv2.addWeighted(copy, 0.8, black_lines, 1, 1)
#     # plt.imshow(lanes)
#     # cv2.waitKey(0)


#     cv2.imshow('Thresh', Thresh)
#     # define q as the exit button
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         break

# # release the video capture object
# cap.release()
# # Closes all the windows currently opened.
# cv2.destroyAllWindows()

if __name__ == "__main__":
    cap = cv2.VideoCapture('pexels-gentayangan-9265063.mp4')
    # Check if camera opened successfully
    if (cap.isOpened()== False):
        print("Error opening video  file")
    # print("hi")
    while(cap.isOpened()):
        ret, frame = cap.read()
#         print("hi")
        copy = np.copy(frame)
        # print(frame.shape)
        plt.figure(figsize=(16,18))

        edges = cv2.Canny(copy,50,150)
#         plt.imshow(edges)
#         break
        isolated = region(edges)


        # cv2.waitKey(0)

        #DRAWING LINES: (order of params) --> region of interest, bin size (P, theta), min intersections needed, placeholder array,
        lines = cv2.HoughLinesP(isolated, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
        averaged_lines = average(copy, lines)
        black_lines = display_lines(copy, averaged_lines)
        #taking wighted sum of original image and lane lines image
        lanes = cv2.addWeighted(copy, 0.8, black_lines, 1, 1)
#         plt.imshow(lanes)
#         break
        cv2.imshow('frame',lanes)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
