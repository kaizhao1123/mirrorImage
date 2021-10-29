# from SlicingResult import GetLengthAndWidth
#
# GetLengthAndWidth('55', 'lg', '1', 'L', 'M')

import cv2 as cv
import numpy as np
from volume import procVolume, displayResult, getVolume, normalizeImage
import os
import sys
import time
import pandas as pd


def main():
    # directory = 'pic/SG-Sept23-33-2'
    directory = 'pic/'
    csv_data = []
    csv_folder_name = ''

    n = 50
    vint = 50

    if (len(sys.argv) > 1):
        numSlice = int(sys.argv[1])
    else:
        numSlice = n
    ########################################################################
    # Main function flow and input image
    ########################################################################
    # get the main view image of the same seed
    for filename in os.listdir(directory):
        if filename.endswith("M.bmp"):
            path = (directory + '/' + filename)
            img = cv.imread(path)
            n = len(filename)
            prefixName = filename[0:n - 5]
            viewName = filename[8:9]

            # set images name for saving
            imgnameForSaving = (directory + '/' + prefixName + 'M-' + str(numSlice) + '.jpg')

            save = True
            display = True
            model = "ellip"  # rect

            # if viewName == 'L':
            #     ratio = 22.525169428    # target : 115/5.1054
            # else:
            #     ratio = 23.308653582  # target: 119/5.1054
            #     # ratio = 21.937556313    # mirror: 112/5.1054

            # pixel and mm ratio
            # ratio = 23.954372623574  # 36.9496855346
            ratio = 110/5.1054  # 21.5458142359 #
            # ratio = 23.308653582    # R
            # ratio = 22.525169428    # L

            # set the HSV range for main view for wheat
            HSV_lower = np.array([0, 0, vint])
            HSV_upper = np.array([255, 255, 255])

            normalizeImage(img, HSV_lower, HSV_upper)
            length_target, length_mirror, width, height, rectArray_target, rectArray_mirror = procVolume(ratio,
                                                                                                         HSV_lower,
                                                                                                         HSV_upper,
                                                                                                         img,
                                                                                                         numSlice,
                                                                                                         display)
            volume_target = getVolume(length_target, length_mirror, rectArray_target, rectArray_mirror, ratio, model)
            # print("volume is", volume)
            print(prefixName + "M.bmp, %d, v=%.4f, l=%.4f, w=%.4f, h=%.4f" % (numSlice, volume_target,
                                                                              length_target, width, height))
            if display:
                displayResult(length_target, length_mirror, width, height, volume_target, img, imgnameForSaving, save,
                              display)
    # save the result to excel file
    if csv_folder_name == '':
        csv_folder_name = imgnameForSaving.split("/")[1]
    csv_imgname = imgnameForSaving.split("/")[2]

    csv_data.append(
        [csv_imgname, float("{:.4f}".format(length_target)), float("{:.4f}".format(width)),
         float("{:.4f}".format(height)), float("{:.4f}".format(volume_target))])

    df = pd.DataFrame(csv_data,
                      columns=['Img Name', 'Length (mm)', 'Width (mm)', 'Height (mm)','Volume (mm\u00b3)'])
    #df.to_excel("output.xlsx", sheet_name=csv_folder_name, index=False)
    return


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("Total time: --- %s seconds ---" % (time.time() - start_time) + "\n")

