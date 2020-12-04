import cv2
import numpy as np
import imutils
import urllib.request
from threading import Thread

from pyglet.resource import file
from playsound import playsound
import multiprocessing
class Detection(object):
    THRESHOLD = 1500

    def __init__(self, image):
        print(image)
        self.previous_gray = image

    def get_active_cell(self, image):
        # obtain motion between previous and current image
        current_gray = image
        print(self.previous_gray)
        delta = cv2.absdiff(self.previous_gray, current_gray)
        threshold_image = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
        # debug
        cv2.imshow('OpenCV Detection', image)
        cv2.waitKey(10)
        cv2.imshow('OpenCV Detection1', current_gray)
        cv2.waitKey(10)
        cv2.imshow('OpenCV Detection2', threshold_image)
        cv2.waitKey(10)

        # store current image
        self.previous_gray = current_gray

        # set cell width
        height, width = threshold_image.shape[:2]
        cell_width = width / 7

        # store motion level for each cell
        cells = np.array([0, 0, 0, 0, 0, 0, 0])
        cells[0] = cv2.countNonZero(threshold_image[0:height, 0:int(cell_width)])
        cells[1] = cv2.countNonZero(threshold_image[0:height, int(cell_width):int(cell_width) * 2])
        cells[2] = cv2.countNonZero(threshold_image[0:height, int(cell_width) * 2:int(cell_width) * 3])
        cells[3] = cv2.countNonZero(threshold_image[0:height, int(cell_width) * 3:int(cell_width) * 4])
        cells[4] = cv2.countNonZero(threshold_image[0:height, int(cell_width) * 4:int(cell_width) * 5])
        cells[5] = cv2.countNonZero(threshold_image[0:height, int(cell_width) * 5:int(cell_width) * 6])
        cells[6] = cv2.countNonZero(threshold_image[0:height, int(cell_width) * 6:width])

        # obtain the most active cell
        top_cell = np.argmax(cells)

        # return the most active cell, if threshold met
        if (cells[top_cell] >= self.THRESHOLD):
            print("worked")
            return top_cell
        else:
            print("not")
            return None

# musical notes (C, D, E, F, G, A, B)
NOTES = [262, 294, 330, 350, 393, 441, 494]
piano_cascade = cv2.CascadeClassifier('cascade.xml') # We load the cascade for the face.


p=cv2.imread("/home/lotfi/face_recognition/Computer_Vision_A_Z_Template_Folder/Module 1 - Face Recognition/p/images.png")
def detect(gray,frame):

    roi_gray=[]
    pianos=piano_cascade.detectMultiScale(gray,1.01,1,minSize=(180,180))
    for (x,y,w,h) in pianos:

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        roi_gray=gray[y:y+h,x:x+w]
        

        
    return frame,roi_gray

cap=cv2.VideoCapture(1)

#cv2.imshow("a",detect(cv2.cvtColor(p,cv2.COLOR_BGR2GRAY),p))
switch=True
while True:
    _, frame = cap.read()
    frame,r=detect(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),frame)
    cv2.imshow("a",frame)
    if len(r)>0:
        
        detection = Detection(r)
    # get current frame from webcam
    

    # use motion detection to get active cell
        cell = detection.get_active_cell(r)
        if cell == None: continue
    
        # if switch on, play note
        if switch:
            print(cell)
            if cell==6:
                p = multiprocessing.Process(target=playsound,
                                            args=('/home/lotfi/PycharmProjects/piano_interpreter/sounds/A4.mp3',))
                p.start()
                cv2.waitKey(500)
                p.terminate()
            if cell == 5:
                p = multiprocessing.Process(target=playsound,
                                            args=('/home/lotfi/PycharmProjects/piano_interpreter/sounds/B4.mp3',))
                p.start()
                cv2.waitKey(500)
                p.terminate()
            if cell == 4:
                p = multiprocessing.Process(target=playsound,
                                            args=('/home/lotfi/PycharmProjects/piano_interpreter/sounds/C4.mp3',))
                p.start()
                cv2.waitKey(500)
                p.terminate()
            if cell == 3:
                p = multiprocessing.Process(target=playsound,
                                            args=('/home/lotfi/PycharmProjects/piano_interpreter/sounds/D4.mp3',))
                p.start()
                cv2.waitKey(500)
                p.terminate()
            if cell == 2:
                p = multiprocessing.Process(target=playsound,
                                            args=('/home/lotfi/PycharmProjects/piano_interpreter/sounds/E5.mp3',))
                p.start()
                cv2.waitKey(500)
                p.terminate()
            if cell == 1:
                p = multiprocessing.Process(target=playsound,
                                            args=('/home/lotfi/PycharmProjects/piano_interpreter/sounds/F4.mp3',))
                p.start()
                cv2.waitKey(500)
                p.terminate()
    #cv2.imshow("Piano found", frame)


    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
# Destroy all the windows
cv2.destroyAllWindows()

class Webcam:

    def __init__(self):
        self.video_capture = cv2.VideoCapture(1)
        self.current_frame = self.video_capture.read()[1]

    # create thread for capturing images
    def start(self):
        Thread(target=self._update_frame, args=()).start()

    def _update_frame(self):
        while (True):
            self.current_frame = self.video_capture.read()[1]

    # get the current frame
    def get_current_frame(self):
        return self.current_frame

class Detection_piano:
    kernel_blur = 15
    sueil = 14
    surface = 3000

    def start(self):

        cap = cv2.VideoCapture(1)


        orignal = cv2.imread("/home/lotfi/PycharmProjects/piano_interpreter/piano_datasets/4.jpg")
        cv2.imshow("r",orignal)
        orignal = cv2.cvtColor(orignal, cv2.COLOR_BGR2GRAY)
        orignal = cv2.GaussianBlur(orignal, (self.kernel_blur, self.kernel_blur), 0)
        print(orignal.shape)
        kernal_dilate = np.ones((5, 5), np.uint8)
        while True:
            ret, frame = cap.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (self.kernel_blur, self.kernel_blur), 0)
            print(gray.shape)
            mask = cv2.absdiff(orignal[0:460][0:630], gray[0:460][0:630])

            mask = cv2.threshold(mask, self.sueil, 255, cv2.THRESH_BINARY)[1]
            cv2.imshow("Mask", mask)
            mask = cv2.dilate(mask, kernal_dilate, iterations=3)
            countours, nad = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
            frame_countour = frame.copy()
            print(frame_countour)
            for c in countours:
                if cv2.contourArea(c) < self.surface:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.imshow("frame", frame)
            orignal = gray
            # cv2.putText(frame,"sss",(10,30),cv2.FONT_HERSHEY_COMPLEX,cv2.C)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


class Detection_Mouvement:
    kernel_blur=15
    sueil=14
    surface=3000
    def start(self):

        cap=cv2.VideoCapture(1)

        ret,orignal=cap.read()
        orignal=cv2.cvtColor(orignal,cv2.COLOR_BGR2GRAY)
        orignal=cv2.GaussianBlur(orignal,(self.kernel_blur,self.kernel_blur),0)
        kernal_dilate=np.ones((5,5),np.uint8)
        while True:
            ret, frame = cap.read()

            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            gray=cv2.GaussianBlur(gray,(self.kernel_blur,self.kernel_blur),0)
            mask=cv2.absdiff(orignal,gray)

            mask=cv2.threshold(mask,self.sueil,255,cv2.THRESH_BINARY)[1]
            cv2.imshow("Mask", mask)
            mask=cv2.dilate(mask,kernal_dilate,iterations=3)
            countours,nad=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]
            frame_countour=frame.copy()
            print(frame_countour)
            for c in countours:
                cv2.drawContours(frame_countour,[c],0,(0,255,255),5)
                if cv2.contourArea(c)<self.surface:
                    continue
                print(c)
                x,y,w,h=cv2.boundingRect(c)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.imshow("countour",(frame_countour))
            cv2.imshow("frame", frame)
            orignal=gray
            #cv2.putText(frame,"sss",(10,30),cv2.FONT_HERSHEY_COMPLEX,cv2.C)
            key=cv2.waitKey(30)&0xFF
            if key==ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()




# initialise detection with first webcam frame


