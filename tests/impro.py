import data.image_process as impro

url = './Final_Training/Images/00000/00000_00024.ppm'

import cv2

image = cv2.imread(url)
image = impro.transform_image(image, 40, 0, 0)
cv2.imshow('aaa', image)
cv2.waitKey()
cv2.destroyAllWindows()