import numpy as np
import cv2
from matplotlib import pyplot as plt
from time import monotonic
from time import sleep


def find_all_four_corners(mask):
    try:
        # Initiate FAST object with default values
        orb = cv2.ORB_create()        # Initiate SIFT detector

        # find the keypoints and descriptors with SIFT
        kp, _ = orb.detectAndCompute(mask, None)
        orb = None
        #img2 = cv2.drawKeypoints(img, kp, outImage=None, color=(255,0,0))
        #cv2.imshow('fast_true.png',img2)

        #mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        #mask = cv2.line(mask,(0, int(mask.shape[0]/2)),(mask.shape[1], int(mask.shape[0]/2)),(255,0,0),5)
        #mask = cv2.line(mask,(int(mask.shape[1]/2), 0),(int(mask.shape[1]/2), mask.shape[0]),(255,0,0),5)

        center = ( int(mask.shape[1]/2), int(mask.shape[0]/2) ) # x, y
        top_left = []
        top_right = []
        bottom_left = []
        bottom_right = []
        for keypoint in kp:
            if keypoint.pt[0] < center[0] - center[0]/2 and keypoint.pt[1] < center[1] - center[1]/2:
                top_left.append(keypoint)
            elif keypoint.pt[0] < center[0] - center[0]/2 and keypoint.pt[1] > center[1] + center[1]/2:
                bottom_left.append(keypoint)
            elif keypoint.pt[0] > center[0] + center[0]/2 and keypoint.pt[1] < center[1] - center[1]/2:
                top_right.append(keypoint)
            elif keypoint.pt[0] > center[0] + center[0]/2 and keypoint.pt[1] > center[1] + center[1]/2:
                bottom_right.append(keypoint)

        most_top_left = top_left[0]
        for point in top_left:
            if point.pt[0] < most_top_left.pt[0] and point.pt[1] < most_top_left.pt[1]:
                most_top_left = point

        most_bottom_left = bottom_left[0]
        for point in bottom_left:
            if point.pt[0] < most_bottom_left.pt[0] and point.pt[1] > most_bottom_left.pt[1]:
                most_bottom_left = point

        most_top_right = top_right[0]
        for point in top_right:
            if point.pt[0] > most_top_right.pt[0] and point.pt[1] < most_top_right.pt[1]:
                most_top_right = point

        most_bottom_right = bottom_right[0]
        for point in bottom_right:
            if point.pt[0] > most_bottom_right.pt[0] and point.pt[1] > most_bottom_right.pt[1]:
                most_bottom_right = point

        top_left = []
        top_right = []
        bottom_left = []
        bottom_right = []
        #kp = []
        # return kp
        return [most_top_left, most_top_right, most_bottom_left, most_bottom_right]
    except:
        return [cv2.KeyPoint(x=0, y=0, _size = 1), cv2.KeyPoint(x=0, y=0, _size = 1), cv2.KeyPoint(x=0, y=0, _size = 1), cv2.KeyPoint(x=0, y=0, _size = 1)]


# works in darkness
def find_screen_contour(image):
    contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    areaArray = []
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        areaArray.append(area)

    sorteddata = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)
    #second_biggest = sorteddata[1][1]
    # determine the most extreme points along the contour
    #moving_top = tuple(second_biggest[second_biggest[:, :, 0].argmin()][0])
    mask = np.zeros(image.shape, np.uint8)

    try:
        biggest = sorteddata[0][1]
        cv2.drawContours(mask, [biggest], -1, 255, -1)
        #cv2.drawContours(mask, [second_biggest], -1, 255, -1)

        # this is necessary to find all four contours
        t = (21, 21)
        s = (7, 7)
        mask = cv2.GaussianBlur(mask, t, 0)
    except:
        print('rip')
    return mask


# works in darkness
def findcenterofpen(image):
    # this is necessary to find all four contours
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    t = (21, 21)
    s = (7, 7)
    image = cv2.GaussianBlur(image, s, 0)

    contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    areaArray = []
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        areaArray.append(area)

    sorteddata = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)

    mask = np.zeros(image.shape, np.uint8)

    try:
        biggest = sorteddata[0][1]

        #cv2.drawContours(mask, contours, -1, 255, -1)

        # center of pen light
        M = cv2.moments(biggest)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        #cv2.circle(mask, (cX, cY), 7, (255, 255, 255), -1)

        return cX, cY
    except:
        return 0, 0


'''
def resizer(patient):
    max_dimension = max(patient.shape)
    scale = 700/max_dimension
    lilshit2 = cv2.resize(patient, None, fx=scale, fy=scale)
    return lilshit2
'''

def canny(patient):
    thresh = cv2.threshold(patient, 60, 255, cv2.THRESH_BINARY)[1] #33.9 for one.png and 60 for two.png
    return thresh

def find_pen(image):
    # Convert the image to HSV colour space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define a range for blue color
    hsv_l = np.array([0,0,176])
    hsv_h = np.array([98,255,255])
    # Find blue pixels in the image
    #
    # cv2.inRange will create a mask (binary array) where the 1 values
    # are blue pixels and 0 values are any other colour out of the blue
    # range defined by hsv_l and hsv_h
    maskerer = cv2.inRange(hsv, hsv_l, hsv_h)
    res = cv2.bitwise_and(image,image, mask= maskerer)
    return res

def works_once(image):
    imageer = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imageer = cv2.cvtColor(imageer, cv2.COLOR_RGB2GRAY)
    #h, s, v1 = cv2.split(lmfao)
    #cv2.imshow('lmfao',v1)
    # cv2.imshow('image',image)
    imageer = canny(imageer)
    mask = find_screen_contour(imageer)

    results = find_all_four_corners(mask)

    #cv2.imshow('mask',mask)
    return results, mask
    #print('fps: ' + str(round(monotonic() * 1000) - start))


#start = round(monotonic() * 1000)
# you must find a way to only  transfer lower resolution images from phone
image_file = "C:/Users/BenBouali/Desktop/tactile screen/three.png"
#image = cv2.imread(image_file)

onetime = True
results = None
mask = None
cap = cv2.VideoCapture('C:/Users/BenBouali/Desktop/tactile screen/der.mp4')
while(cap.isOpened()):

    ret, frame = cap.read()
    if ret == True:
        image = frame
        res = find_pen(image)
        cX, cY = findcenterofpen(res)
        #cv2.imshow('res',resser)

        if onetime:
            onetime = False
            results, mask = works_once(image)

        if results!=None and mask.all()!=None:
            results.append(cv2.KeyPoint(x=cX, y=cY, _size = 1)) # center of pen
            mask = cv2.drawKeypoints(mask, results, outImage=None, color=(255,0,0))
            results.remove(results[4])

            cv2.imshow('mask',mask)
            cv2.imshow('image',image)
            sleep(0.05q)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()

# while True:
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         cv2.destroyAllWindows()
#         break
