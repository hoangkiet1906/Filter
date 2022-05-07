import cv2
import numpy as np
import dlib


def empty():
    pass
cv2.namedWindow("BGR")
cv2.resizeWindow("BGR",360,130)
cv2.createTrackbar("Blue",'BGR',0,255,empty)
cv2.createTrackbar("Green",'BGR',0,255,empty)
cv2.createTrackbar("Red",'BGR',0,255,empty)

def createBox(img,point,scale=5,to=False):
    mask = np.zeros_like(img)
    mask = cv2.fillPoly(mask,[point],(255,255,255))
    #cv2.imshow("mask", mask)
    img = cv2.bitwise_and(img,mask)
    #cv2.imshow("mask", img)
    bbox = cv2.boundingRect(point)
    x,y,w,h = bbox
    imgCrop = img[y:y+h,x:x+w]
    imgCrop = cv2.resize(imgCrop,(0,0),None,scale,scale)
    if to:
        return mask
    else:
        return imgCrop


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    img = cv2.imread("chin.jpg")
    img = cv2.resize(img,(0,0),None,0.5,0.5)
    imgE = img.copy()
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = detector(imgGray)
    for f in faces:
        x1,y1 = f.left(),f.top()
        x2,y2 = f.right(),f.bottom()
        #print(x1,y1,x2,y2)
        landmark = predictor(imgGray,f)

    points = []

    #cv2.rectangle(imgE,(x1,y1),(x2,y2),(255,0,0),2)
    for i in range(68):
        x = landmark.part(i).x
        y = landmark.part(i).y
        points.append([x,y])
        #cv2.circle(imgE,(x,y),2,(0,255,0),cv2.FILLED)
        #cv2.putText(imgE,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,0,255),1)

    points = np.array(points)
    #imgL_Eye = createBox(imgE,points[36:42])
    imgLip = createBox(img,points[48:61],to=True)
    imgEye = createBox(img,points[36:41],to=True)
    imgLipColor = np.zeros_like(imgLip)
    imgEyeColor = np.zeros_like(imgEye)
    b = cv2.getTrackbarPos("Blue",'BGR')
    g = cv2.getTrackbarPos("Green",'BGR')
    r = cv2.getTrackbarPos("Red",'BGR')
    #print(b,g,r)
    imgLipColor[:] = int(b),int(g),int(r)
    imgEyeColor[:] = int(b),int(g),int(r)
    imgLipColor = cv2.bitwise_and(imgLip,imgLipColor)
    imgEyeColor = cv2.bitwise_and(imgEye,imgEyeColor)

    imgLipColor = cv2.GaussianBlur(imgLipColor,(7,7),10)
    imgEyeColor = cv2.GaussianBlur(imgEyeColor,(7,7),10)
    imgLipColor = cv2.addWeighted(imgE,1,imgLipColor,0.4,0)
    imgEyeColor = cv2.addWeighted(imgE,1,imgEyeColor,0.4,0)
    #cv2.imshow("L_Eye", imgLip)
    #cv2.imshow("Filter", imgE)
    #cv2.imshow("BGR", imgLipColor)
    cv2.imshow("BGR", imgEyeColor)
    #cv2.imshow("Eye", imgEye)

    cv2.waitKey(1)