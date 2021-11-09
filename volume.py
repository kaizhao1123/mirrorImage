import cv2 as cv
import numpy as np
import math

delta = 5


# decrease the mirror to the same size with target
def normalizeImage(img, vint):

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # change to gray image
    # Global threshold segmentation,  to binary image. (Otsu)
    res, dst = cv.threshold(gray, vint, 255, 0)  # 0,255 cv2.THRESH_OTSU
    element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))  # Morphological denoising
    dst = cv.morphologyEx(dst, cv.MORPH_OPEN, element)  # Open operation denoising
    contours, hierarchy = cv.findContours(dst, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # Contour detection function

    count = 0
    result = []
    twoConts = []
    for cont in contours:
        area = cv.contourArea(cont)  # Calculate the area of the enclosing shape
        if area < 2000:
            continue
        rect = cv.minAreaRect(cont)

        # draw the min area rect
        # box = cv.boxPoints(rect)
        # box = np.int0(box)
        # cv.drawContours(img, [box], 0, (0, 0, 255), 2)  # red
        twoConts.append(cont)
        result.append(rect[1])

        count += 1
        cv.putText(img, str(count), (math.ceil(rect[0][0]), math.ceil(rect[0][1])), cv.FONT_HERSHEY_COMPLEX,
                   0.6, (255, 0, 0), 1)  # blue
    x, y, w, h = cv.boundingRect(twoConts[0])   # target
    x_m, y_m, w_m, h_m = cv.boundingRect(twoConts[1])   # mirror
    r = w_m / w
    mirrorImg = img[y_m:y_m+h_m, x_m:x_m+w_m]
    newW = w   # w_m / r
    newH = math.ceil(h_m / r)
    dim = (newW, newH)
    newTarget = cv.resize(mirrorImg, dim, interpolation=cv.INTER_AREA)
    edge = int(20)
    halfEdge = int(edge / 2)
    template = np.zeros([newH + edge, newW + edge, 3], dtype=np.uint8)
    template[halfEdge: newH + halfEdge, halfEdge: newW + halfEdge] = newTarget

    img[y_m-halfEdge: y_m+halfEdge + newH, x-halfEdge: x+halfEdge + newW] = template
    cv.imshow('View', img)
    cv.waitKey()




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


def newDrawLine(contour, img):
    Y_list = getAllpoint(contour)
    for i in range(0, len(Y_list), 2):

        pointX = Y_list[i][0]
        pointY_top = Y_list[i][1]
        pointY_bottom = Y_list[i][2]

        # left line
        cv.line(img, (pointX, pointY_top), (pointX, pointY_bottom), (0, 0, 255), 1)


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


def findNextUpper(dict, index, mid):
    while index <= list(dict)[-1]:
        if index in dict:
            vList = dict.get(index)
            if suitForSlicing(vList, mid):
                if vList[0] < vList[-1]:
                    return index, vList[0]
                else:
                    return index, vList[-1]
            else:
                if vList[0] <= mid:
                    return index, vList[0]
                else:
                    index += 1
        else:
            index += 1
    return -1, -1

def findNextLower(dict, index, mid):
    while index <= list(dict)[-1]:
        if index in dict:
            vList = dict.get(index)
            if suitForSlicing(vList, mid):
                if vList[0] < vList[-1]:
                    return index, vList[-1]
                else:
                    return index, vList[0]
            else:
                if vList[0] >= mid:
                    return index, vList[0]
                else:
                    index += 1
        else:
            index += 1
    return -1, -1


def getAllpoint(contour):
    contour = sorted(contour, key=lambda tup: tup[0][0])  # sort the contour, from left point to right point

    # initial a dictionary
    contour_dict = {}  # create dictionary to record the contour point with the same X value, X: [Y1, Y2}...
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
    print(contour_dict)

    # find the left, right point, and mid_y
    left_point, right_point = newGetLeftRight(contour_dict)
    y_mid = getMidPoint(left_point, right_point, left_point[0])
    startPoint_X = left_point[0]
    endPoint_X = right_point[0]
    print(y_mid)
    ####
    # create a list, to record each pixel's Y-value, which used to slice. for example: (X, topY, bottomY)
    ####
    Y_list = []

    # add the first point (the most left) into list
    previousPoint_X = startPoint_X
    previousPoint_listValue = contour_dict.get(previousPoint_X)
    if len(previousPoint_listValue) > 1:
        previousTop = findTop(previousPoint_listValue, y_mid)
        previousBottom = findBottom(previousPoint_listValue, y_mid)
    else:
        previousTop = y_mid
        previousBottom = y_mid
    Y_list.append((startPoint_X, previousTop, previousBottom))

    # add others into list
    current_X = startPoint_X + 1
    while current_X <= endPoint_X:
        if current_X in contour_dict:
            current_listValue = contour_dict.get(current_X)  # list all y-values from all point with the same x-value
            current_mid = getMidPoint(left_point, right_point, current_X)
            topY = findTop(current_listValue, current_mid)
            bottomY = findBottom(current_listValue, current_mid)

            if not suitForSlicing(current_listValue, current_mid):  # or current_X == endPoint_X:
                if topY == bottomY:  # the most right point
                    print("right point")
                elif topY != -1:
                    bottomY = previousBottom
                elif bottomY != -1:
                    topY = previousTop
            Y_list.append((current_X, topY, bottomY))
            previousTop = topY
            previousBottom = bottomY

        else:
            Y_list.append((current_X, previousTop, previousBottom))
        current_X += 1

    # update the list
    size = len(Y_list)
    for i in range(0, size):
        k = Y_list[i][0]
        mid = getMidPoint(left_point, right_point, k)
        if k in contour_dict:
            vList = contour_dict.get(k)
            if suitForSlicing(vList, mid):
                continue
            else:
                originalTopY = findTop(vList, mid)
                originalBottomY = findBottom(vList, mid)

                if originalTopY != -1:
                    nextX, lower = findNextLower(contour_dict, k+1, mid)
                    temp = list(Y_list[i])
                    temp[2] = math.floor((lower - Y_list[i - 1][2]) / (nextX - k) + Y_list[i - 1][2])
                    temp = tuple(temp)
                    Y_list[i] = temp
                elif originalBottomY != -1:
                    nextX, upper = findNextUpper(contour_dict, k + 1, mid)
                    temp = list(Y_list[i])
                    temp[1] = math.floor((upper - Y_list[i - 1][1]) / (nextX - k) + Y_list[i - 1][1])
                    temp = tuple(temp)
                    Y_list[i] = temp
        else:
            nextX_u, upper = findNextUpper(contour_dict, k+1, mid)
            nextX_l, lower = findNextLower(contour_dict, k+1, mid)
            temp = list(Y_list[i])
            temp[1] = math.floor((upper - Y_list[i - 1][1]) / (nextX_u - k) + Y_list[i - 1][1])
            temp[2] = math.floor((lower - Y_list[i - 1][2]) / (nextX_l - k) + Y_list[i - 1][2])
            temp = tuple(temp)
            Y_list[i] = temp

    print(Y_list)
    print("%%%% new %%%%%%")
    return Y_list


def newSlicingRect(contour):
    Y_list = getAllpoint(contour)
    rect_array = []
    for i in range(0, len(Y_list)-1):
        a = (Y_list[i][1] + Y_list[i+1][1]) / 2
        b = (Y_list[i][2] + Y_list[i+1][2]) / 2
        rect_array.append((1, math.floor(b - a)))
    print(rect_array)
    return rect_array

########################################################################
# calculate volume using slicing rectangle box
########################################################################
def getVolume(length_t, length_m, rectArr_t, rectArr_m, ratio, model):
    # scale factor length_target / length_mirror
    # convert unit to mm by divid the ratio
    # V = pi * A * B * h
    r = length_t / length_m
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
def procView(vint, img, num_slice, ratio, display, auto=False):

    # if not auto:
    #     # convert the image to the hsv data format
    #     hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    #     hsv = cv.GaussianBlur(hsv, (3, 3), 10)
    #     hsv = cv.dilate(hsv, (3, 3))
    #     # Threshold the HSV image to get only brown colors
    #     mask = cv.inRange(hsv, HSVlower, HSVupper)
    #     ret, thresh = cv.threshold(mask, 230, 255, 0)
    #     # ret, thresh = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
    #     print("auto")
    # else:
    #     mask = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #     ret, thresh = cv.threshold(mask, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    #
    # contours, hierarchy = cv.findContours(thresh, 1, 2)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # change to gray image
    # Global threshold segmentation,  to binary image. (Otsu)
    res, dst = cv.threshold(gray, vint, 255, 0)  # 0,255 cv2.THRESH_OTSU
    element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))  # Morphological denoising
    dst = cv.morphologyEx(dst, cv.MORPH_OPEN, element)  # Open operation denoising
    contours, hierarchy = cv.findContours(dst, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # Contour detection function

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

        # count += 1
        # cv.putText(img, str(count), (math.ceil(rect[0][0]), math.ceil(rect[0][1])), cv.FONT_HERSHEY_COMPLEX,
        #            0.6, (255, 0, 0), 1)  # blue

    x, y, w, h = cv.boundingRect(twoConts[0])
    # box_mirror = np.array([(x, y), (x, y + h), (x + w, y), (x + w, y + h)])
    # if len(twoConts) > 1:
    #     x, y, w, h = cv.boundingRect(twoConts[1])
    #     box_target = np.array([(x, y), (x, y + h), (x + w, y), (x + w, y + h)])
    # else:
    #     box_target = box_mirror

    if display:

        # drawLine(img, box_mirror, twoConts[0], num_slice)
        newDrawLine(twoConts[0], img)
        cv.drawContours(img, [twoConts[0]], 0, (255, 0, 0), 1)
        if len(twoConts) > 1:
            # drawLine(img, box_target, twoConts[1], num_slice)
            newDrawLine(twoConts[1], img)
            cv.drawContours(img, [twoConts[1]], 0, (255, 0, 0), 1)

    length_t_pixel = result[0][1]
    height_pixel = result[0][0]
    if length_t_pixel < height_pixel:
        height_pixel, length_t_pixel = length_t_pixel, height_pixel
    length_t = length_t_pixel / ratio
    height = height_pixel / ratio
    # rectArray_m = slicingRect(box_mirror, twoConts[0], num_slice)
    rectArray_t = newSlicingRect(twoConts[0])

    #######################
    # newSlicingRect(twoConts[0])
    # newSlicingRect(twoConts[1])
    ########################

    if len(twoConts) > 1:
        length_m = result[1][1]
        width = result[1][0]
        if length_m < width:
            width, length_m = length_m, width
        r = length_m / length_t_pixel  # length_m / length_t
        length_m = length_m / ratio / r
        width = width / ratio / r
        # rectArray_t = slicingRect(box_target, twoConts[1], num_slice)
        rectArray_m = newSlicingRect(twoConts[1])
    else:
        length_m = length_t
        width = height
        rectArray_m = rectArray_t

    return length_t, length_m, width, height, rectArray_t, rectArray_m


########################################################################
# display the result
########################################################################
def displayResult(length_t, length_m, width, height, volume, img, imgnameForSaving, save, display):
    # construct the string show on the images

    r = float(length_t / length_m)  # the ratio of length_target / length_mirror

    width = width * r
    length = "Length is  %.3f" % length_t + " mm"
    width = "Width is  %.3f" % width + " mm"
    height = "Height is  %.3f" % height + " mm"
    volume = "Volume is  %.3f" % volume + " mm^3"

    cv.putText(img, length, (10, 60), 0, 0.5, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(img, height, (10, 90), 0, 0.5, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(img, volume, (10, 150), 0, 0.5, (255, 255, 255), 2, cv.LINE_AA)

    cv.putText(img, width, (10, 120), 0,0.5, (255, 255, 255), 2, cv.LINE_AA)
    if save:
        cv.imwrite(imgnameForSaving, img)
    if display:
        cv.imshow('View', img)
        cv.waitKey()
    return
def newProcView(contour, img, ratio, display):

    rect = cv.minAreaRect(contour)
    length = math.floor(rect[1][0])
    width = math.floor(rect[1][1])
    length = length / ratio
    width = width / ratio
    if length < width:
        width, length = length, width
    if display:
        drawLine(contour, img)
        cv.drawContours(img, [contour], 0, (255, 0, 0), 1)
    rectArray = slicingRect(contour)
    return length, width, rectArray


def procVolume(ratio, vint, img, num_slice, display, auto=False):
    # Calculate the length and width using procView function for side view
    length_target, length_mirror, width, height, rectArray_target, rectArray_mirror = procView(vint, img, num_slice,
                                                                                               ratio, display, auto)

    return length_target, length_mirror, width, height, rectArray_target, rectArray_mirror



