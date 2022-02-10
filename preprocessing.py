import cv2
import os 
from natsort import os_sorted

data_folder_path = "data"
folder = os_sorted(os.listdir(data_folder_path))

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
for frame in folder:
    frame = os.path.join(data_folder_path,frame)
    img = cv2.imread(frame)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        faces = img[y:y + h, x:x + w]
        cv2.imshow("face",faces)
        cv2.imwrite(frame, faces)

