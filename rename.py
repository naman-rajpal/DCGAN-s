import os 
from natsort import os_sorted

data_folder_path = "images"
folder = os_sorted(os.listdir(data_folder_path))
i=0
for frame in folder:
    frame = os.path.join(data_folder_path,frame)
    name = str(i) +".jpg"
    new_frame = os.path.join(data_folder_path,name)
    os.rename(frame, new_frame)
    i+=1