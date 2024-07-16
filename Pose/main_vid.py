import pandas as pd
import numpy as np
import cv2

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

w = 350 # Uygun bir sayı
h = 350

model = cv2.dnn.readNetFromTensorflow('graph_opt.pb') # Ön eğitimli bir model
# Deep Natural Network

capture = cv2.VideoCapture('pose_video.mp4')

while cv2.waitKey(1)<0:
    hasFrame, img = capture.read() # Bir sonraki kareye bakıyor, okunabiliyorsa hasFrame 1, değilse 0
    img = cv2.resize(img, (600, 338)) # Hızlı olsun diye tekrardan boyutlandırılıyor
    if not hasFrame:
        # Yeni bir kare yoksa alışacak
        cv2.waitKey()
        break
    
        
    f_w = img.shape[1] # (h, w, d)
    f_h = img.shape[0] # (yükseklik, genişlik, kanal sayısı)

    # görselin modele girdi olarak tanımlanması

    model.setInput(cv2.dnn.blobFromImage(img, 1.0, (w, h), (127.5, 127.5, 127.5), swapRB = True, crop = False)) # görsel, scale factor ( ölçek faktörü - istediğimiz sayı ile çarpıyor yani boyutlandırıyor-), shape, mean parametresi ( rgb tarzında bir tuple -her kanaldan çıkan ortalama değer için-, hepsini 1 ve -1 arasında atıyor)
    # blob bynary large object
    # swapRB BGR den RGB ye dönüştürüyor
    # !!! crop?

    detect = model.forward() # Tahminleri kaydettik

    detect = detect[:, :19, :, :] # İlk 19 kanal
    # Heatmap yöntemiyle buluyor. Pikseller yoğun olunca önemli olduğunu anlıyor
    # Sadece bodypart olduğu için diğer parametreler boş
    
    assert(len(BODY_PARTS) == detect.shape[1]) # Kanal saysısının eşit olup olmadığını ölçüyor
    # assert koşulu doğrulamaya çalışıyor
    # Eğer uyuşmuyorlarsa bize bir hata döndürüyor
    
    # heatmap işleme
    
    # x, y değişkenlerini kaydetme koordinat
    
    points = []
    
    for i in range(len(BODY_PARTS)):
        
        heatMap = detect[0, i, :, :] # Tensordan bir dilim alıyor
        # (N, C, H, W) fotoğraf sayısı, kanal sayısı (body partları, zaten 36. satırda yaptık), bütün satır ve sütunlar
        
        _, conf, _, point = cv2.minMaxLoc(heatMap) # minumun ve maksimum yoğunluk noktalarını buluyoruz
        # Alt çizgilere (minımumlar) ihtiyacımız yok o yüzden _
        x = (f_w * point[0])/detect.shape[3]
        y = (f_h*point[1])/detect.shape[2]
        points.append((int(x), int(y)) if conf > 0.2 else None) # Virgüllü bir sayı olsa bile virgülü kaldırıyoruz
        
    # vücut parçalarını bağlama
    
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)
        
        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]
        
        if points[idFrom] and points[idTo]:
            
            cv2.line(img, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.ellipse(img, points[idFrom], (3, 3), 0, 0, 360, (0,0,255), cv2.FILLED) # BOY, KALINLIK, DOLDURULMUŞ
            cv2.ellipse(img, points[idTo], (3, 3), 0, 0, 360, (0,0,255), cv2.FILLED)
            
    cv2.imshow('Pose Detection', img)
    # cv2.waitKey(0) Burayı koymuyoruz ki bir video oluşsun

        
        
        
 

