import cv2

face_model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

capture = cv2.VideoCapture(0) # Anlık Görüntü Oluşturacağız

while True:
    
    hasFrame, img = capture.read()
    if not hasFrame:
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detected_faces = face_model.detectMultiScale(gray, 1.1, 4)
    
    for (x,y, w, h) in detected_faces:
        
        cv2.rectangle(img, (x, y), (x+w, y+h), (150, 255, 0), 1)
        
    cv2.imshow('Face Detection (LIVE)', img)
    
    # Her milisaniyede
    
    if cv2.waitKey(1) and 0xFF == ord('q'): # ord ile de yapılabiliyor 
        break
    
capture.release()
cv2.destroyAllWindows()

# 1. Ödev: Face Detection işlemini hereketli bir görüntü işleminde yapın
# 2. Ödev: Pose Detection işlemini kendi kameramızda yapacağız
        
