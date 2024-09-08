import cv2
import numpy as np
import time

cap = cv2.VideoCapture('video.mp4')

min_width_react = 80
min_height_react = 80

count_line_position = 550

algo = cv2.bgsegm.createBackgroundSubtractorMOG()

def centre_handle(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

detect = []
offset = 6  
counter_in = 0  
counter_out = 0  

while True:
    ret, frame1 = cap.read()
    if not ret:
        break  
    
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    counterShape, _ = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (255, 127, 0), 3)
    
    for c in counterShape:
        (x, y, w, h) = cv2.boundingRect(c)
        validate_counter = (w >= min_width_react) and (h >= min_height_react)
        if not validate_counter:
            continue
        
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame1, "Vehicle", (x, y - 20), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 244, 0), 2)       
                    
        center = centre_handle(x, y, w, h)
        detect.append(center)
        cv2.circle(frame1, center, 4, (0, 0, 255), -1)
        
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        for (cx, cy) in detect:
            if cy < count_line_position and cy > (count_line_position - offset):
                counter_in += 1
                detect.remove((cx, cy))
                print(f"{timestamp} - VEHICLE IN: {counter_in}")
            elif cy > count_line_position and cy < (count_line_position + offset):
                counter_out += 1
                detect.remove((cx, cy))
                print(f"{timestamp} - VEHICLE OUT: {counter_out}")
        

    cv2.putText(frame1, "VEHICLES IN: " + str(counter_in), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.putText(frame1, "VEHICLES OUT: " + str(counter_out), (450, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)


    
    cv2.imshow('Detecter',dilatada)
    
    cv2.imshow('DL Project Vehicle Counter',frame1)# To display the 'Video Original' frame
    

    if cv2.waitKey(1) == 13:
        break
    
cv2.destroyAllWindows()
cap.release()
