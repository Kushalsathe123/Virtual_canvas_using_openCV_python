import cv2
import mediapipe as mp
import numpy as np
import math


class hand:
    
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, minTrackCon=0.5):
        
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.detectionCon, self.minTrackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.fingers = []
        self.lmList = []

    def findHands(self, img, draw=True):
        
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        
        xList = []
        yList = []
        bbox = []
        bboxInfo =[]
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                px, py = int(lm.x * w), int(lm.y * h)
                xList.append(px)
                yList.append(py)
                self.lmList.append([px, py])
                if draw:
                    cv2.circle(img, (px, py), 5, (255, 0, 255), cv2.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            boxW, boxH = xmax - xmin, ymax - ymin
            bbox = xmin, ymin, boxW, boxH
            cx, cy = bbox[0] + (bbox[2] // 2), bbox[1] + (bbox[3] // 2)
            bboxInfo = {"id": id, "bbox": bbox,"center": (cx, cy)}

            if draw:
                cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),(bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),(0, 255, 0), 2)

        return self.lmList, bboxInfo

    def fingersUp(self):
        
        if self.results.multi_hand_landmarks:
            myHandType = self.handType()
            fingers = []
            # Thumb
            if myHandType == "Right":
                if self.lmList[self.tipIds[0]][0] > self.lmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if self.lmList[self.tipIds[0]][0] < self.lmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # 4 Fingers
            for id in range(1, 5):
                if self.lmList[self.tipIds[id]][1] < self.lmList[self.tipIds[id] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers

    

    def handType(self):
        
        if self.results.multi_hand_landmarks:
            if self.lmList[17][0] < self.lmList[5][0]:
                return "Right"
            else:
                return "Left"


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
font = cv2.FONT_HERSHEY_SIMPLEX
dec=hand(maxHands=1,detectionCon=0.7)
xp=0
yp=0
imdraw= np.zeros((720, 1280,3),np.uint8)
while True:
    success,img= cap.read()
    img = cv2.flip(img,1)
    img = dec.findHands(img)
    lmlist,bbox=dec.findPosition(img)
    if lmlist:
        x1,y1 = lmlist[8]
        x2,y2 = lmlist[12]
        fingers = dec.fingersUp()
        if fingers==[0,1,0,0,0]:
            if xp==0 and yp ==0:
                xp,yp =x1,y1         
            cv2.line(img,(xp,yp),(x1,y1),(0,0,255),5,cv2.FILLED)
            cv2.line(imdraw,(xp,yp),(x1,y1),(0,0,255),5,cv2.FILLED)
            cv2.putText(img, 'hello', (bbox[0],bbox[0]), font, 1, (255,0,255), 2, cv2.LINE_AA)
            xp,yp=x1,y1
        
        elif fingers==[0,1,1,0,0]:
            cv2.circle(img, (x1, y1), 20, (0, 0, 0), cv2.FILLED)           
            cv2.circle(imdraw, (x1, y1), 20, (0, 0, 0), cv2.FILLED)
            
        else:
            xp=0
            yp=0
    
    gray= cv2.cvtColor(imdraw,cv2.COLOR_BGR2GRAY)
    _,inv = cv2.threshold(gray,50,255,cv2.THRESH_BINARY_INV)
    inv=cv2.cvtColor(inv,cv2.COLOR_GRAY2BGR)

    img = cv2.bitwise_and(img,inv)
    img = cv2.bitwise_or(img,imdraw)
    
    cv2.imshow("Image",img)
    #cv2.imshow("draw",imdraw)
    #cv2.imshow("Inv",inv)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
        
  #yay this is fun

