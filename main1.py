import cv2

cap01=cv2.VideoCapture("/dev/video2")
ret0, frame0 = cap01.read()
print(frame0)
cv2.imshow('frame', frame0)
cv2.waitKey()