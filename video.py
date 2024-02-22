import cv2

vid = cv2.VideoCapture("smile-detection/smile.mp4")
smile_cascade = cv2.CascadeClassifier('smile-detection/smile.xml')
face_cascade = cv2.CascadeClassifier('smile-detection/frontalface.xml')

while True:
    ret, frame = vid.read()
    frame = cv2.resize(frame, (640,480))
    if ret == False:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for x,y,w,h in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
        
        roi = frame[y:y+h,x:x+w]
        roi_gray = gray[y:y+h,x:x+w]
        
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 3)
        
        for sx,sy,sw,sh in smiles:
            cv2.rectangle(roi, (sx,sy),(sx+sw, sy+sh), (0,255,0), 2)
    
    cv2.imshow("frame", frame)
    cv2.waitKey(20)