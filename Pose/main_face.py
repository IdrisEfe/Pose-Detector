import cv2
import numpy as np

face_model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('faces.jpg')

# Genellikle gri tonlama kullanılır yüz işlemlerinde

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

detected_faces = face_model.detectMultiScale(gray, 1.1, 4) # Her tarama penceresinde yüzde 10 atırıyoruz (1.1)
# Dört minimum komşu sayısı, 1-5 arasındadır, ne kadar büyükse o kadar hassas

# Yüzleri kareler ile çizeceğiz

for (x,y, w, h) in detected_faces:
    
    cv2.rectangle(img, (x, y), (x+w, y+h), (150, 255, 0), 1) # Sol üst köşe ve sağ alt köşe
    
cv2.imshow('Face Detection', img)
cv2.waitKey(0)
