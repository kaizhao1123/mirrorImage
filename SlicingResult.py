import cv2
import math
import numpy as np

def drawSepLine(image, contour):
    l_p = []
    r_p = []
    xmin = np.amin(contour, axis=0)[0][0]
    xmax = np.amax(contour, axis=0)[0][0]
    for p in contour:
        if p[0][0] < xmin + 5:  # set up the gap to 5
            l_p.append(p[0][1])
        if p[0][0] > xmax - 5:
            r_p.append(p[0][1])

    x, y, w, h = cv2.boundingRect(contour)
    # box = cv.boxPoints(rect)
    box = np.array([(x, y), (x, y + h), (x + w, y), (x + w, y + h)])

    l = (np.amin(box, axis=0)[0], int(sum(l_p) / len(l_p)))
    r = (np.amax(box, axis=0)[0], int(sum(r_p) / len(r_p)))
    # center line
    cv2.line(image, (l[0], l[1]), (r[0], r[1]), (255, 255, 255), 1)
    # M = cv.moments(countour)
    # # print(M)
    # cX = int(M["m10"] / M["m00"])
    # cY = int(M["m01"] / M["m00"])
    # cv.circle(image, (cX, cY), 2, (255, 255, 255), -1)
   # return (l, r)


"""
seedType: for example, 11/22
size: for example, lg/md/sm/xl
num: for example, 1/2/3/4/5
view: for example, L/R
angle: for example, A/M/O
"""
def GetLengthAndWidth(seedID, size, num, view, angle):
    # imgname = 'pic/' + ("%04d" % imageNum) + '.bmp'
    # imgname = 'pic/55-lg-1-R-M.bmp'
    imgname = 'pic/' + seedID + '-' + size + '-' + num + '-' + view + '-' + angle + '.bmp'
    # imgname = 'pic/R REF.bmp'

    img = cv2.imread(imgname)  # input image

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # change to gray image

    res, dst = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # Morphological denoising
    dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, element)  # Open operation denoising
    contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)  # Contour detection function
    # cv2.drawContours(dst, contours, -1, (120, 0, 0), 2)  # Draw contour

    count = 0
    result = []
    for cont in contours:
        area = cv2.contourArea(cont)  # Calculate the area of the enclosing shape
        if area < 1000:
            continue
        rect = cv2.minAreaRect(cont)
        print(rect)
        # draw the min area rect
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img, [box], 0, (0, 0, 255), 1)
        result.append(rect[1])
        # Write the number in the upper left corner of the rectangle
        count += 1
        cv2.putText(img, str(count), (math.ceil(rect[0][0]), math.ceil(rect[0][1])), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 0), 1)

        drawSepLine(img, cont)


    targetLengthPixel = math.ceil(result[1][1])
    targetWidthPixel = math.ceil(result[1][0])
    mirrorLengthPixel = math.ceil(result[0][1])
    mirrorWidthPixel = math.ceil(result[0][0])

    print("{} side target(pixel): length:{} width:{}".format(view, targetLengthPixel, targetWidthPixel))
    print("{} side mirror(pixel): length:{} width:{}".format(view, mirrorLengthPixel, mirrorWidthPixel))

    ratio = 5.1054 / 110 # 117.785
    # ratio2 = 5.1054 / 109 # 109.5
    r = targetLengthPixel / mirrorLengthPixel

    targetLength = targetLengthPixel * ratio / r
    targetWidth = targetWidthPixel * ratio / r
    mirrorLength = mirrorLengthPixel * ratio
    mirrorWidth = mirrorWidthPixel * ratio

    print("{} side target: length:{} mm, width:{} mm".format(view, targetLength, targetWidth))
    print("{} side mirror: length:{} mm, width:{} mm".format(view, mirrorLength, mirrorWidth))



    # show original image, gray image
    cv2.namedWindow("original", 1)
    cv2.imshow('original', img)
    cv2.namedWindow("dst", 1)
    cv2.imshow("dst", dst)
    cv2.waitKey()




