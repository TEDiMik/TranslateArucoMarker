import httplib2
import math
#import cv2
#import numpy as np
#import cv2, PIL
#from cv2 import aruco
#import matplotlib.pyplot as plt
#import matplotlib as mpl

import numpy
import cv2, PIL
from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

import numpy as np
import cv2
import glob

'''
h = httplib2.Http('.cache')
response, content = h.request('https://stepik.org/media/attachments/lesson/284187/example_2.jpg')
out = open('img.jpg', 'wb')
out.write(content)
out.close()
'''
def calculateDistance(x1,y1,x2,y2):  
     dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
     return dist  



frame = cv2.imread("Sample/img2.jpg")

#----------------------------------------------------------------------
'''
distCoeffs=[0.0189223469433419, -0.0206788674793396,0.003225513523750, 0.001510000668961]
cameraMatrix=[[600.8293, 0, 330.2756],[0, 601.3519, 225.9791],[0, 0, 1.0000]]


img = frame
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
aruco_dict = cv2.aruco_Dictionary.get(aruco.DICT_4X4_1000)
parameters = cv2.aruco_DetectorParameters.create()
corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
cv2.aruco.drawDetectedMarkers(img,corners,ids,)
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(corners, 0.045, cameraMatrix, distCoeffs)
cv2.aruco.drawAxis(img, cameraMatrix, distCoeffs, rvecs, tvecs, 0.1)
'''
#----------------------------------------------------------------------

#frame = cv2.rotate(frame,cv2.ROTATE_90_CLOCKWISE)

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



aruco_dict = aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)

parameters = aruco.DetectorParameters_create()
#parameters.adaptiveThreshConstant = 7


corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

font = cv2.FONT_HERSHEY_SIMPLEX





if np.all(ids != None):

    

    frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)

    plt.figure()
    plt.imshow(frame_markers)
    for i in range(len(ids)):
        c = corners[i][0]
        plt.plot([c[:, 0].mean()], [c[:, 1].mean()], "o", label = "id={0}".format(ids[i]))
    plt.legend()
    plt.show()

    
    #corners[id marker][NotUnderstand][angle][SmallNotUnderstand]
    x = int (corners[1][0][0][0])
    y = int (corners[1][0][0][1])

    x1 = int (corners[2][0][0][0])
    y1 = int (corners[2][0][0][1])

    dist = calculateDistance(x,y,x1,y1)
    print('DISTANCE: ', dist)

    frameCOlor = cv2.line(frame_markers,(x,y),(0,0),(255,0,0),5)
    frameCOlor = cv2.line(frame_markers,(x1,y1),(0,0),(255,0,0),5)
    print('x =', x)
    print('y =', y)

    
    plt.imshow(frameCOlor)
    
    plt.show()



else:
    # code to show 'No Ids' when no markers are found
    cv2.putText(frame, "No Ids", (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)
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

'''
def MarkerDetection4x4(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco_Dictionary.get(aruco.DICT_4X4_1000)
    parameters = cv2.aruco_DetectorParameters.create()
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    cv2.aruco.drawDetectedMarkers(img,corners,ids,)
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(corners, 0.045, cameraMatrix, distCoeffs)
    cv2.aruco.drawAxis(img, cameraMatrix, distCoeff, rvecs, tvecs, 0.1)
'''


print('Все')

