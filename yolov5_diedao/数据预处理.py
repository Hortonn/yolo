import os
import cv2

dataDir = "VOCdevkit/images/val1/"
saveDir = "VOCdevkit/images/val/"
if not os.path.exists(saveDir):
    os.makedirs(saveDir)

for one_pic in os.listdir(dataDir):
    one_path = dataDir + one_pic
    one_img = cv2.imread(one_path)
    new_path = saveDir + one_pic
    cv2.imwrite(new_path, one_img)