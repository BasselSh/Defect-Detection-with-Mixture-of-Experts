import cv2
import os
root = 'cam_out_scratches'
files = sorted(os.listdir(root), key=lambda s: int(s.split('.')[0]))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('box_removed_extra.avi', fourcc, 15.0, (300, 300))
for file in files:
    img = cv2.imread(f"{root}/{file}")
    img = cv2.resize(img, (300,300))
    print(file)
    out.write(img)