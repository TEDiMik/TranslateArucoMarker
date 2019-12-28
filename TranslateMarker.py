import httplib2
import cv2
from cv2 import aruco
import matplotlib.pyplot as plt
import numpy as np



# I'm tired of giving concerts

#Loading image by link
'''
h = httplib2.Http('.cache')
response, content = h.request('https://stepik.org/media/attachments/lesson/284187/example_2.jpg')
out = open('img.jpg', 'wb')
out.write(content)
out.close()
'''






frame = cv2.imread("Sample/img6.jpg")


hsv_min = np.array((-7, 123, 116))
hsv_max = np.array((13, 198, 196))
color_yellow = (0,255,255)



#Convert 
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#Find red point 
thresh = cv2.inRange(hsv, hsv_min, hsv_max)

#Find Aruco marcer, White Square 4X4
aruco_dict = aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)

#In documentation write what need
parameters = aruco.DetectorParameters_create()

#angle markers/find markers
corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

font = cv2.FONT_HERSHEY_SIMPLEX





#Detect red point objects
moments = cv2.moments(thresh, 1)
sum_y = moments['m01']
sum_x = moments['m10']
sum_pixel = moments['m00']

if sum_pixel > 5:
    #Coordinate red point object
    x_RedPoint = int(sum_x / sum_pixel)
    y_RedPoint = int(sum_y / sum_pixel)

    frameCOlor1 = cv2.line(frame,(x_RedPoint,y_RedPoint),(0,0),(255,0,0),5)
    plt.imshow(frameCOlor1)
    #Paint circle in coordinate red point
    cv2.circle(frame, (x_RedPoint, y_RedPoint), 10, (0, 0, 255), -1)
    # write coordinate
    cv2.putText(frame, "%d-%d" % (x_RedPoint, y_RedPoint), (x_RedPoint + 50, y_RedPoint - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color_yellow, 2)


# if markers finding 
if np.all(ids != None):

    #Короч берем центр координат макреа и центр точки обьекта из большего вычитаем меньшее и получаем координаты обьекта

    frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)

    x = int (corners[1][0][0][0]) + int((int(corners[0][0][2][0]) - int (corners[0][0][0][0])) / 2)
    y = int (corners[1][0][0][1]) + int((int(corners[0][0][2][1]) - int (corners[0][0][0][1])) / 2)

    plt.figure()
    plt.imshow(frame_markers)
    for i in range(len(ids)):
        c = corners[i][0]
        if ids[i] == 101:
            cv2.putText(frame_markers, "%d-%d" % (x, y), (x + 50, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color_yellow, 2)
        
        #plt.plot([c[:, 0].mean()], [c[:, 1].mean()], "o", label = "id=".format(ids[i]))
    plt.legend()
    plt.show()

    
    #corners[id marker][NotUnderstand][angle][SmallNotUnderstand]

    #На координатной (0,0)
    #Вычесление центра маркера
    x0 = int (corners[1][0][0][0]) + int((int(corners[0][0][2][0]) - int (corners[0][0][0][0])) / 2)
    y0 = int (corners[1][0][0][1]) + int((int(corners[0][0][2][1]) - int (corners[0][0][0][1])) / 2)


    #Ось X
    x2 = int (corners[2][0][0][0])
    y2 = int (corners[2][0][0][1])

    #Ось Y
    x1 = int (corners[1][0][0][0])
    y1 = int (corners[1][0][0][1])

    print('Координаты точки: ', 'x = {0}; y = {1}'.format(x_RedPoint,y_RedPoint))
    print('Координаты маркера: ', 'x = {0}; y = {1}'.format(x0,y0))


    #dist = calculateDistance(x,y,x1,y1) #In old code
    #print('DISTANCE: ', dist)

    frameCOlor = cv2.line(frame_markers,(x0,y0),(0,0),(255,0,0),5)
    #frameCOlor = cv2.line(frame_markers,(x0,y0),(x1,y1),(255,0,0),5)
    print('x =', x)
    print('y =', y)

    
    plt.imshow(frameCOlor)
    
    plt.show()



else:
    # else write No markers
    cv2.putText(frame, "No Ids", (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)
    plt.imshow(frame)
    plt.show()




#old code
#
'''
def calculateDistance(x1,y1,x2,y2):  
     dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
     return dist  
'''
# #----------------------------------------------------------------------
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










#End programm
#For my self
print('Все')

