import cv2
import numpy as np
import imutils
import time
import pyautogui
#blank_image = np.zeros((height,width,3), np.uint8)

def resizer(patient):
    max_dimension = max(patient.shape)
    scale = 700/max_dimension
    lilshit2 = cv2.resize(patient, None, fx=scale, fy=scale)
    return lilshit2

def canny(patient):
    gray = cv2.cvtColor(patient, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7,7), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
    return thresh

def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]
    head = 100
    polygons = np.array([
    [(head, 0), (head, height), (width, height), (width, 0)]
    ])
    mask = np.zeros(image.shape, np.uint8) #mask1 = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked = cv2.bitwise_and(image, mask)
    return masked

def main(frame):
    image = canny(frame)
    contours, _ = cv2.findContours(image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    areaArray = []
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        areaArray.append(area)

    sorteddata = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)
    biggest = sorteddata[0][1]
    second_biggest = sorteddata[1][1]
    # determine the most extreme points along the contour
    moving_top = tuple(second_biggest[second_biggest[:, :, 0].argmin()][0])
    static_top = tuple(biggest[biggest[:, :, 0].argmin()][0])
    moving_y, moving_x = moving_top
    _, static_x = static_top #min_second_value=380 #max_second_value=400
    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, [biggest], -1, 255, -1)
    cv2.drawContours(mask, [second_biggest], -1, 255, -1)

    resized_and_cropped = region_of_interest(resizer(mask))
    resized_and_cropped_and_rotated = imutils.rotate_bound(resized_and_cropped, 90)
    gap = 25
    if moving_x > static_x-gap and moving_x < static_x+gap:
        cv2.putText(resized_and_cropped_and_rotated, "tap", (193, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), lineType=cv2.LINE_AA)

        # the real shit

        # the real shit

    return resized_and_cropped_and_rotated


vid = cv2.VideoCapture("vid.mp4")
while(vid.isOpened()):
    _, frames = vid.read()

    result = main(frames)
    cv2.imshow("result", result)
    time.sleep(0.055) # 0.055 - enhance me daddy
    if cv2.waitKey(1) == 27:
        vid.release()
        cv2.destroyAllWindows()
        break
