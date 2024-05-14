"""
GROUP 10: 
Member : 
 1/ Danh Truong Son - 20110394
 2/ Nguyen Duc Huy - 20145449
 3/ Nguyen Trung Nguyen - 20110388
"""
from sys import path
from tkinter import*
from PIL import Image,ImageTk
import os
import cv2
import numpy as np
from time import strftime
from datetime import datetime
import face_recognition

path = 'AttendanceImage'
images = []
classNames = []

class Attendance:
        


    def __init__(self,root):
        self.root=root
        self.root.geometry("1366x768+0+0")
        self.root.title("Group 10 - Member: Danh Truong Son - 20110394 / Nguyen Trung Nguyen - 20110388 / Nguyen Duc Huy - 20145449")
        #Folder Image Path

        # Automic get the name of images
        myList = os.listdir(path)
        print(myList)
        for cl in myList:
            curImg = cv2.imread(f'{path}/{cl}')
            images .append(curImg)
            classNames.append(os.path.splitext(cl)[0]) # ==> Get The Name (Without the file type)
        print(classNames)
        # This part is image labels setting start 
        # first header image  
        img=Image.open(r"C:/Users/Son/Documents/Study/Deep_Learning/FINAL/Image/banner.jpg")
        img=img.resize((1366,130),Image.ANTIALIAS)
        self.photoimg=ImageTk.PhotoImage(img)

        # set image as lable
        f_lb1 = Label(self.root,image=self.photoimg)
        f_lb1.place(x=0,y=0,width=1366,height=130)

        # backgorund image 
        bg1=Image.open(r"C:/Users/Son/Documents/Study/Deep_Learning/FINAL/Image/bg3.jpg")
        bg1=bg1.resize((1366,768),Image.ANTIALIAS)
        self.photobg1=ImageTk.PhotoImage(bg1)

        # set image as lable
        bg_img = Label(self.root,image=self.photobg1)
        bg_img.place(x=0,y=130,width=1366,height=768)


        #title section
        title_lb1 = Label(bg_img,text="Welcome to Face Recognition Pannel",font=("verdana",30,"bold"),bg="white",fg="navyblue")
        title_lb1.place(x=0,y=0,width=1366,height=45)

        # Create buttons below the section 
        # ------------------------------------------------------------------------------------------------------------------- 
        # Training button 1
        std_img_btn=Image.open(r"C:/Users/Son/Documents/Study/Deep_Learning/FINAL/Image/f_det.jpg")
        std_img_btn=std_img_btn.resize((180,180),Image.ANTIALIAS)
        self.std_img1=ImageTk.PhotoImage(std_img_btn)

        std_b1 = Button(bg_img,command=self.face_recog,image=self.std_img1,cursor="hand2")
        std_b1.place(x=600,y=170,width=180,height=180)

        std_b1_1 = Button(bg_img,command=self.face_recog,text="Face Detector",cursor="hand2",font=("tahoma",15,"bold"),bg="darkblue",fg="white")
        std_b1_1.place(x=600,y=350,width=180,height=45)
        
    #=====================Attendance===================
    def findEncodings(self,images):
        encodeList = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        return encodeList
    def markAttendance(self,name):
        with open('Attendance.csv','r+') as f:
            myDataList = f.readlines()
            nameList = []
            for line in myDataList:
                entry = line.split(',')
                nameList.append(entry[0])
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{dtString}')


    #================face recognition==================
    def face_recog(self):
        encodeListKnown = self.findEncodings(images)
        print('Encoding Complete')

        videoCap=cv2.VideoCapture(0)

        while True:
            success, img = videoCap.read()
            #img = captureScreen()
            imgS = cv2.resize(img,(0,0),None,0.25,0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
            
            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
            
            for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
                # matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
                # Check which images in folder compare with the picture capture through camera having the lowest faceDis then return that person
                print(faceDis)
                matchIndex = np.argmin(faceDis)
                
                if faceDis[matchIndex]< 0.50:
                    name = classNames[matchIndex].upper()
                    print(name)
                    y1,x2,y2,x1 = faceLoc
                    y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
                    #We have scale down the image at the beggining so multipy 4 to return the corect location
                    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                    cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                    self.markAttendance(name)
                else: 
                    name = 'Unknown'
                    print(name)
                    y1,x2,y2,x1 = faceLoc
                    y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
                    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                    cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cv2.imshow('Webcam',img)
        videoCap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root=Tk()
    obj=Attendance(root)
    root.mainloop()
