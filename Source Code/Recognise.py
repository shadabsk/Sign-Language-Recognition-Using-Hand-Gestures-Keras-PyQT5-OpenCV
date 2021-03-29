__author__ = 'Shadab Shaikh, Obaid Kazi, Rupesh poudel'
__SourcerepoLink__ = 'https://github.com/rrupeshh/Simple-Sign-Language-Detector'

import cv2
import numpy as np
import os

def nothing(x):
    pass

image_x, image_y = 64,64

from keras.models import load_model
classifier = load_model('ASLModel.h5')

#original = cv2.imread("5.png")
fileEntry=[]
for file in os.listdir("SampleGestures"):
    if file.endswith(".png"):
    	fileEntry.append(file)

def imgprocessing():
  image_to_compare = cv2.imread("./SampleGestures/space.png")
  original = cv2.imread("1.png")
  sift = cv2.xfeatures2d.SIFT_create()
  kp_1, desc_1 = sift.detectAndCompute(original, None)
  kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)
   
  index_params = dict(algorithm=0, trees=5)
  search_params = dict()
  flann = cv2.FlannBasedMatcher(index_params, search_params)
    
  matches = flann.knnMatch(desc_1, desc_2, k=2)
   
  good_points = []
  ratio = 0.6
  for m, n in matches:
      if m.distance < ratio*n.distance:
          good_points.append(m)
  #print(len(good_points))
  fin=len(kp_1) -len(kp_2)
  print(abs(fin))
  if(abs(fin)<=2):
    return 'space'
    
  result = cv2.drawMatches(original, kp_1, image_to_compare, kp_2, good_points, None)
   
  #cv2.imshow("result", result)
  #cv2.imshow("Original", original)
  #cv2.imshow("Duplicate", image_to_compare)
  #cv2.destroyAllWindows()

def predictor():
       import numpy as np
       from keras.preprocessing import image
       test_image = image.load_img('1.png', target_size=(64, 64))
       test_image = image.img_to_array(test_image)
       test_image = np.expand_dims(test_image, axis = 0)
       result = classifier.predict(test_image)
       for i in range(len(fileEntry)):
	       image_to_compare = cv2.imread("./SampleGestures/"+fileEntry[i])
	       original = cv2.imread("1.png")
	       sift = cv2.xfeatures2d.SIFT_create()
	       kp_1, desc_1 = sift.detectAndCompute(original, None)
	       kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)
	       
	       index_params = dict(algorithm=0, trees=5)
	       search_params = dict()
	       flann = cv2.FlannBasedMatcher(index_params, search_params)
	        
	       matches = flann.knnMatch(desc_1, desc_2, k=2)
	       
	       good_points = []
	       ratio = 0.6
	       for m, n in matches:
	              if m.distance < ratio*n.distance:
	                     good_points.append(m)
	       print(fileEntry[i])
	       if(abs(len(good_points)+len(matches))>20):
	       	gesname=fileEntry[i]
	       	gesname=gesname.replace('.png','')
	        return gesname
       #print(good_points)
       #if(abs(fin)<=2):
             # return 'space'
       for i in range(26):
              if result[0][i] == 1:
	             return chr(ord('A') + i)  
       

   

cam = cv2.VideoCapture(0)

cv2.namedWindow("Trackbars")

cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

cv2.namedWindow("ASL Recognition")

img_text = ''
img_text1 = ''
#z5=1
while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame,1)
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")


    img = cv2.rectangle(frame, (425,100),(625,300), (0,255,0), thickness=2, lineType=8, shift=0)

    lower_blue = np.array([l_h, l_s, l_v])
    upper_blue = np.array([u_h, u_s, u_v])
    imcrop = img[102:298, 427:623]
    hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    cv2.putText(frame, img_text, (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0))
    cv2.putText(frame, img_text1, (80, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0))
    cv2.imshow("ASL Recognition", frame)
    cv2.imshow("mask", mask)
        
    img_name = "1.png"
    save_img = cv2.resize(mask, (image_x, image_y))
    cv2.imwrite(img_name, save_img)
    img_text = predictor()

    if cv2.waitKey(1) == 27:
        break

   # img_text1 = imgprocessing()

   


cam.release()
cv2.destroyAllWindows()
