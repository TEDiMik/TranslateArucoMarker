import httplib2
#import cv2
#import numpy as np
#import cv2, PIL
#from cv2 import aruco
#import matplotlib.pyplot as plt
#import matplotlib as mpl

import numpy as np
import cv2, PIL
from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd


'''
h = httplib2.Http('.cache')
response, content = h.request('https://stepik.org/media/attachments/lesson/284187/example_2.jpg')
out = open('img.jpg', 'wb')
out.write(content)
out.close()
'''

frame = cv2.imread("index.png")


gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()
parameters.adaptiveThreshConstant = 10
corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

font = cv2.FONT_HERSHEY_SIMPLEX


if np.all(ids != None):

        # estimate pose of each marker and return the values
        # rvet and tvec-different from camera coefficients
    rvec, tvec ,_ = aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, dist)
        #(rvec-tvec).any() # get rid of that nasty numpy value array error

    for i in range(0, ids.size):
        # draw axis for the aruco markers
        aruco.drawAxis(frame, mtx, dist, rvec[i], tvec[i], 0.1)

        # draw a square around the markers
    aruco.drawDetectedMarkers(frame, corners)


        # code to show ids of the marker found
    strg = ''
    for i in range(0, ids.size):
        strg += str(ids[i][0])+', '

    cv2.putText(frame, "Id: " + strg, (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)


else:
        # code to show 'No Ids' when no markers are found
    cv2.putText(frame, "No Ids", (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)



plt.figure()
plt.imshow(frame)
plt.show()
'''
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters =  aruco.DetectorParameters_create()
corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)


frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)

aruco.drawDetectedMarkers(frame.copy(), corners)

plt.imshow(frame_markers)



plt.show()
'''



print('Все')

