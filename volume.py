import cv2 as cv
import numpy as np
import math

delta = 5


# decrease the target to the same size with mirror
def normalizeImage(img, HSVlower, HSVupper):

    # convert the image to the hsv data format
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    hsv = cv.GaussianBlur(hsv, (3, 3), 10)
    hsv = cv.dilate(hsv, (3, 3))
    # Threshold the HSV image to get only brown colors
    mask = cv.inRange(hsv, HSVlower, HSVupper)
    ret, thresh = cv.threshold(mask, 127, 255, 0)
    # ret, thresh = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
    # else:
    #     mask = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #     ret, thresh = cv.threshold(mask, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    contours, hierarchy = cv.findContours(thresh, 1, 2)

    count = 0
    result = []
    twoConts = []
    for cont in contours:
        area = cv.contourArea(cont)  # Calculate the area of the enclosing shape
        if area < 2000:
            continue
        rect = cv.minAreaRect(cont)

        # draw the min area rect
        box = cv.boxPoints(rect)
        box = np.int0(box)
        #cv.drawContours(img, [box], 0, (0, 0, 255), 2)  # red
        twoConts.append(cont)
        result.append(rect[1])

        #count += 1
        #cv.putText(img, str(count), (math.ceil(rect[0][0]), math.ceil(rect[0][1])), cv.FONT_HERSHEY_COMPLEX,
        #           0.6, (255, 0, 0), 1)  # blue
    x, y, w, h = cv.boundingRect(twoConts[0])   # mirror
    x_t, y_t, w_t, h_t = cv.boundingRect(twoConts[1])   # target
    r = w_t / w
    targetImg = img[y_t:y_t+h_t, x_t:x_t+w_t]
    newW = w   # w_t / r
    newH = math.ceil(h_t / r)
    dim = (newW, newH)
    newTarget = cv.resize(targetImg, dim, interpolation=cv.INTER_AREA)
    template = np.zeros([newH + 30, newW + 30, 3], dtype=np.uint8)
    template[15: newH + 15, 15: newW + 15] = newTarget

    img[y_t-15: y_t+15 + newH, x-15: x+15 + newW] = template
    # cv.imshow('View', img)
    # cv.waitKey()




########################################################################
# draw seperation line function and return left and right most point
########################################################################
def drawSepLine(image, box, countour):
    l_p = []
    r_p = []
    xmin = np.amin(countour, axis=0)[0][0]
    xmax = np.amax(countour, axis=0)[0][0]
    for p in countour:
        if p[0][0] < xmin + delta:
            l_p.append(p[0][1])
        if p[0][0] > xmax - delta:
            r_p.append(p[0][1])

    l = (np.amin(box, axis=0)[0], int(sum(l_p) / len(l_p)))
    r = (np.amax(box, axis=0)[0], int(sum(r_p) / len(r_p)))
    # center line
    # cv.line(image, (l[0], l[1]), (r[0], r[1]), (255, 255, 255), 1)
    return (l, r)


########################################################################
# get left and right most point
########################################################################
def getLeftRight(contour):
    l_p = []
    r_p = []
    xmin = np.amin(contour, axis=0)[0][0]
    xmax = np.amax(contour, axis=0)[0][0]
    for p in contour:
        if p[0][0] < xmin + delta:
            l_p.append(p[0][1])
        if p[0][0] > xmax - delta:
            r_p.append(p[0][1])
    l = (xmin, int(sum(l_p) / len(l_p)))
    r = (xmax, int(sum(r_p) / len(r_p)))
    return l, r


def newGetLeftRight(contour_dict):
    listKey = list(contour_dict)
    leftKey = listKey[0]
    leftValue = contour_dict.get(leftKey)
    left_Y = int(sum(leftValue) / len(leftValue))
    leftPoint = (leftKey, left_Y)

    rightKey = listKey[-1]
    rightValue = contour_dict.get(rightKey)
    right_Y = int(sum(rightValue) / len(rightValue))
    rightPoint = (rightKey, right_Y)

    return leftPoint, rightPoint


########################################################################
# getMidPoint function return the mid point
########################################################################
def getMidPoint(pointl, pointr, x):
    a = float((pointr[1] - pointl[1]) / (pointr[0] - pointl[0]))
    b = pointl[1] - a * pointl[0]
    y = int(a * x + b)
    return y


########################################################################
# drawLine function for the small slicing piece
# based on the number of slice n
########################################################################
def drawLine(image, box, contour, n):
    x = int(np.amin(box, axis=0)[0])
    x_right = int(np.amax(box, axis=0)[0])
    # seg = int((np.amax(box, axis=0)[0]- x )/ n)
    # hard code to 0.25mm slice 0.25mm * 46.3040446 = 11.57601115
    # seg = 11
    seg = (x_right - x) / float(n)
    seg2 = seg / 2.0
    x = x + seg2
    left_point, right_point = drawSepLine(image, box, contour)
    y_mid = getMidPoint(left_point, right_point, x)
    recent_low = [y_mid]
    recent_high = [y_mid]
    while x <= x_right:
        low = []
        high = []
        for pair in contour:
            if x - seg2 <= pair[0][0] <= x + seg2:
                y_mid = getMidPoint(left_point, right_point, pair[0][0])
                if pair[0][1] >= y_mid:
                    low.append(pair[0][1])
                else:
                    high.append(pair[0][1])
        if len(low) == 0:
            y_lower = recent_low[-1]
        else:
            y_lower = sum(low) / len(low)
            recent_low.append(y_lower)
        if len(high) == 0:
            y_upper = recent_high[-1]
        else:
            y_upper = sum(high) / len(high)
            recent_high.append(y_upper)
        y = y_upper
        line = y_lower - y_upper
        x1 = x - seg / 2.0
        ix1seg = int(np.rint(x1 + seg))
        iyline = int(np.rint(y + line))
        ix1 = int(np.rint(x1))
        iy = int(np.rint(y))
        # bottom line
        cv.line(image, (ix1seg, iyline), (ix1, iyline), (0, 0, 255), 1)

        # upper line
        cv.line(image, (ix1seg, iy), (ix1, iy), (0, 0, 255), 1)

        # left line
        cv.line(image, (ix1, iy), (ix1, iyline), (0, 0, 255), 1)

        # right line
        cv.line(image, (ix1seg, iy), (ix1seg, iyline), (0, 0, 255), 1)
        x = x + seg

    return


def suitForSlicing(list, midPoint):
    hasUpper = False
    hasLower = False
    for ele in list:
        if ele < midPoint:
            hasUpper = True
        else:
            hasLower = True
    if hasUpper is True and hasLower is True:
        return True
    else:
        return False


def findTop(list_V, midPoint):    # Y is the smallest
    if list_V[0] < list_V[-1]:
        if list_V[0] <= midPoint:
            return list_V[0]
        else:
            return -1
    else:
        if list_V[-1] <= midPoint:
            return list_V[-1]
        else:
            return -1


def findBottom(list_V, midPoint):    # Y is the largest
    if list_V[0] < list_V[-1]:
        if list_V[-1] >= midPoint:
            return list_V[-1]
        else:
            return -1
    else:
        if list_V[0] >= midPoint:
            return list_V[0]
        else:
            return -1

########################################################################
# get slicing rectangle box and return as np.array
########################################################################
def slicingRect(box, contour, n):
    # start_time = time.time()
    rect_arr = []
    x = np.amin(box, axis=0)[0]
    seg = (np.amax(box, axis=0)[0] - x) / float(n)
    seg2 = seg / 2.0
    x = x + seg2
    # seg = 11.57601115
    left_point, right_point = getLeftRight(contour)
    y_mid = getMidPoint(left_point, right_point, x)
    recent_low = [y_mid]
    recent_high = [y_mid]

    for i in range(0, n):
        low = []
        high = []
        for pair in contour:
            if x - seg2 <= pair[0][0] <= x + seg2:
                y_mid = getMidPoint(left_point, right_point, pair[0][0])
                if pair[0][1] >= y_mid:
                    low.append(pair[0][1])
                else:
                    high.append(pair[0][1])
        if len(low) == 0:
            y_lower = recent_low[-1]
        else:
            y_lower = sum(low) / len(low)
            recent_low.append(y_lower)
        if len(high) == 0:
            y_upper = recent_high[-1]
        else:
            y_upper = sum(high) / len(high)
            recent_high.append(y_upper)
        rect_arr.append((seg, y_lower - y_upper))
        x = x + seg
    return rect_arr



def newSlicingRect(box, contour):
    # start_time = time.time()
    rect_list = []

    contour = sorted(contour, key=lambda tup: tup[0][0])    # sort the contour, from left point to right point

    # initial a dictionary
    contour_dict = {}   # create dictionary to record the contour point with the same X value, X: [Y1, Y2}...
    k = contour[0][0][0]
    v = []
    v.append(contour[0][0][1])
    contour_dict[k] = v
    contour = contour[1:]
    for ele in contour:
        key = ele[0][0]
        value = ele[0][1]
        if key in contour_dict:
            listValue = contour_dict.get(key)
            listValue.append(value)
        else:
            newValue = []
            newValue.append(value)
            contour_dict[key] = newValue

    left_point, right_point = newGetLeftRight(contour_dict)
    y_mid = getMidPoint(left_point, right_point, left_point[0])
    startPoint_X = left_point[0]
    endPoint_X = right_point[0]

    previousPoint_X = startPoint_X
    previousPoint_listValue = contour_dict.get(previousPoint_X)
    if len(previousPoint_listValue) > 1:
        previousTop = findTop(previousPoint_listValue, y_mid)
        previousBottom = findBottom(previousPoint_listValue, y_mid)
    else:
        previousTop = y_mid
        previousBottom = y_mid

    current_X = startPoint_X + 1
    while current_X <= endPoint_X:
        if current_X in contour_dict:
            current_listValue = contour_dict.get(current_X)
            current_mid = getMidPoint(left_point, right_point, current_X)
            rec_width = 1  #current_X - previousPoint_X # in fact, it is a fixed value : 1 (pixel)
            if suitForSlicing(current_listValue, current_mid):  # or current_X == endPoint_X:

                rec_height = math.fabs(current_listValue[-1] - current_listValue[0])

                #rec_height = current_listValue[0]

                # ##########
                # diff = rec_height - previousPoint_height
                # previousPoint_height = rec_height
                # rec_height = rec_height - (diff / 2)

                # #############

                rect_list.append((rec_width, rec_height))
                previousPoint_X = current_X
                previousTop = findTop(current_listValue, current_mid)
                previousBottom = findBottom(current_listValue, current_mid)
            else:
                top = findTop(current_listValue, current_mid)
                bottom = findBottom(current_listValue, current_mid)
                if top != -1:
                    rec_height = previousBottom - top
                elif bottom != -1:
                    rec_height = bottom - previousTop
                elif top == bottom:
                    rec_height = previousBottom - previousTop
                rect_list.append((rec_width, rec_height))

        else:
            rect_list.append(rect_list[-1])
        current_X += 1
    print(rect_list)
    print("%%%%%%%%%%%%%%%%")
    return rect_list

########################################################################
# calculate volume using slicing rectangle box
########################################################################
def getVolume(length_t, length_m, rectArr_t, rectArr_m, ratio, model):
    # scale factor length_target / length_mirror
    # convert unit to mm by divid the ratio
    # V = pi * A * B * h
    r = float(length_t / length_m)
    rectArr_m = np.array(rectArr_m) * r

    if model == "ellip":
        # apply elliptic cylinder volume model here
        pi = 3.14159265
        A = np.array(rectArr_t)[:, 1] / ratio / 2
        B = np.array(rectArr_m)[:, 1] / ratio / 2
        h = np.array(rectArr_t)[:, 0] / ratio
        V = np.sum(pi * np.multiply(np.multiply(A, B), h))
    else:
        # apply rectangle volume model here
        A = np.array(rectArr_t)[:, 1] / ratio
        B = np.array(rectArr_m)[:, 1] / ratio
        h = np.array(rectArr_m)[:, 0] / ratio
        V = np.sum(np.multiply(np.multiply(A, B), h))
    return V


########################################################################
# Main View
########################################################################
def procView(HSVlower, HSVupper, img, num_slice, ratio, display, auto=False):

    if not auto:
        # convert the image to the hsv data format
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        hsv = cv.GaussianBlur(hsv, (3, 3), 10)
        hsv = cv.dilate(hsv, (3, 3))
        # Threshold the HSV image to get only brown colors
        mask = cv.inRange(hsv, HSVlower, HSVupper)
        ret, thresh = cv.threshold(mask, 127, 255, 0)
        # ret, thresh = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
        print("auto")
    else:
        mask = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(mask, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    contours, hierarchy = cv.findContours(thresh, 1, 2)

    count = 0
    result = []
    twoConts = []
    for cont in contours:
        area = cv.contourArea(cont)  # Calculate the area of the enclosing shape
        if area < 2000:
            continue
        rect = cv.minAreaRect(cont)

        # draw the min area rect
        box = cv.boxPoints(rect)
        box = np.int0(box)
        cv.drawContours(img, [box], 0, (0, 0, 255), 2)  # red
        twoConts.append(cont)
        result.append(rect[1])

        count += 1
        cv.putText(img, str(count), (math.ceil(rect[0][0]), math.ceil(rect[0][1])), cv.FONT_HERSHEY_COMPLEX,
                   0.6, (255, 0, 0), 1)  # blue

    x, y, w, h = cv.boundingRect(twoConts[0])
    box_mirror = np.array([(x, y), (x, y + h), (x + w, y), (x + w, y + h)])
    if len(twoConts) > 1:
        x, y, w, h = cv.boundingRect(twoConts[1])
        box_target = np.array([(x, y), (x, y + h), (x + w, y), (x + w, y + h)])
    else:
        box_target = box_mirror

    if display:
        cv.drawContours(img, [twoConts[0]], 0, (255, 0, 0), 2)
        drawLine(img, box_mirror, twoConts[0], num_slice)
        if len(twoConts) > 1:
            cv.drawContours(img, [twoConts[1]], 0, (255, 0, 0), 2)
            drawLine(img, box_target, twoConts[1], num_slice)

    length_m_pixel = result[0][1]
    width_pixel = result[0][0]
    if length_m_pixel < width_pixel:
        width_pixel, length_m_pixel = length_m_pixel, width_pixel
    length_m = length_m_pixel / ratio
    width = width_pixel / ratio
    # rectArray_m = slicingRect(box_mirror, twoConts[0], num_slice)
    rectArray_m = newSlicingRect(box_mirror, twoConts[0])

    #######################
    # newSlicingRect(box, twoConts[0])
    # newSlicingRect(box, twoConts[1])
    ########################

    if len(twoConts) > 1:
        length_t = result[1][1]
        height = result[1][0]
        if length_t < height:
            height, length_t = length_t, height
        r = length_t / length_m_pixel  # length_t / length_m
        length_t = length_t / ratio / r
        height = height / ratio / r
        # rectArray_t = slicingRect(box_target, twoConts[1], num_slice)
        rectArray_t = newSlicingRect(box_target, twoConts[1])
    else:
        length_t = length_m
        height = width
        rectArray_t = rectArray_m

    return length_t, length_m, width, height, rectArray_t, rectArray_m, mask


########################################################################
# display the result
########################################################################
def displayResult(length_t, length_m, width, height, volume, img, imgnameForSaving, save, display):
    # construct the string show on the images

    r = float(length_t / length_m)  # the ratio of length_target / length_mirror

    width = width * r
    length = "Length is  %.2f" % length_t + " mm"
    width = "Width is  %.2f" % width + " mm"
    height = "Height is  %.2f" % height + " mm"
    volume = "Volume is  %.2f" % volume + " mm^3"

    cv.putText(img, length, (10, 60), 0, 1, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(img, height, (10, 90), 0, 1, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(img, volume, (10, 450), 0, 1, (255, 255, 255), 2, cv.LINE_AA)

    cv.putText(img, width, (10, 120), 0, 1, (255, 255, 255), 2, cv.LINE_AA)
    if save:
        cv.imwrite(imgnameForSaving, img)
    if display:
        cv.imshow('View', img)
        cv.waitKey()
    return


def procVolume(ratio, HSV_lower, HSV_upper, img, num_slice, display, auto=False):
    # Calculate the length and width using procView function for side view
    length_target, length_mirror, width, height, rectArray_target, rectArray_mirror, mask_target = procView(HSV_lower,
                                                                                                            HSV_upper,
                                                                                                            img,
                                                                                                            num_slice,
                                                                                                            ratio,
                                                                                                            display,
                                                                                                            auto)

    return length_target, length_mirror, width, height, rectArray_target, rectArray_mirror



