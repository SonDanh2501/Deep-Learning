"""
GROUP 07: 
Member : 
 1/ Dang Thanh Tuyen - 20110412
 2/ Danh Truong Son - 20110394
 3/ Dang Phuoc Truong Tai - 20110396
"""
from tkinter import* 
from tkinter import ttk
from PIL import Image,ImageTk
import os
from subprocess import call

from StudentList import Student
from Help import Helpsupport
from AttendanceProject import Attendance
from EmotionDetector import EmotionDetector
from Photos import Photos

class Main:
    def __init__(self,root):
        self.root=root
        self.root.title("Group 10 - Member: Danh Truong Son - 20110394 / Nguyen Trung Nguyen - 20110388 / Nguyen Duc Huy - 20145449")
        self.root.geometry("1366x768+0+0")

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
        title_lb1 = Label(bg_img,text="Manage Student With Face Detector",font=("verdana",30,"bold"),bg="white",fg="navyblue")
        title_lb1.place(x=0,y=0,width=1366,height=45)

        # Create buttons below the section 
        # ------------------------------------------------------------------------------------------------------------------- 
        # Button 1: Student List
        std_img_btn=Image.open(r"C:/Users/Son/Documents/Study/Deep_Learning/FINAL/Image/std1.jpg")
        std_img_btn=std_img_btn.resize((180,180),Image.ANTIALIAS)
        self.std_img1=ImageTk.PhotoImage(std_img_btn)

        std_b1 = Button(bg_img,command=self.student_pannels,image=self.std_img1,cursor="hand2")
        std_b1.place(x=370,y=100,width=180,height=180)

        std_b1_1 = Button(bg_img,command=self.student_pannels,text="Student Pannel",cursor="hand2",font=("times new roman",15,"bold"),bg="darkblue",fg="white")
        std_b1_1.place(x=370,y=280,width=180,height=45)
        # Button 2: Emotion Detector
        det_img_btn=Image.open(r"C:/Users/Son/Documents/Study/Deep_Learning/FINAL/Image/f_det.jpg")
        det_img_btn=det_img_btn.resize((180,180),Image.ANTIALIAS)
        self.det_img1=ImageTk.PhotoImage(det_img_btn)

        det_b1 = Button(bg_img,command=self.emotiondetector,image=self.det_img1,cursor="hand2",)
        det_b1.place(x=600,y=100,width=180,height=180)

        det_b1_1 = Button(bg_img,command=self.emotiondetector,text="Emotion Detector",cursor="hand2",font=("times new roman",15,"bold"),bg="darkblue",fg="white")
        det_b1_1.place(x=600,y=280,width=180,height=45)
        # Button 3: Attendance
        att_img_btn=Image.open(r"C:/Users/Son/Documents/Study/Deep_Learning/FINAL/Image/att.jpg")
        att_img_btn=att_img_btn.resize((180,180),Image.ANTIALIAS)
        self.att_img1=ImageTk.PhotoImage(att_img_btn)

        att_b1 = Button(bg_img,command=self.attendance,image=self.att_img1,cursor="hand2",)
        att_b1.place(x=830,y=100,width=180,height=180)

        att_b1_1 = Button(bg_img,command=self.attendance,text="Attendance",cursor="hand2",font=("times new roman",15,"bold"),bg="darkblue",fg="white")
        att_b1_1.place(x=830,y=280,width=180,height=45)
        # Button 4: Photo
        pho_img_btn=Image.open(r"C:/Users/Son/Documents/Study/Deep_Learning/FINAL/Image/photo.jpg")
        pho_img_btn=pho_img_btn.resize((180,180),Image.ANTIALIAS)
        self.pho_img1=ImageTk.PhotoImage(pho_img_btn)

        pho_b1 = Button(bg_img,command=self.photos,image=self.pho_img1,cursor="hand2",)
        pho_b1.place(x=600,y=350,width=180,height=180)

        pho_b1_1 = Button(bg_img,command=self.photos,text="Photos",cursor="hand2",font=("times new roman",15,"bold"),bg="darkblue",fg="white")
        pho_b1_1.place(x=600,y=510,width=180,height=45)
        # Button 5: Help
        hlp_img_btn=Image.open(r"C:/Users/Son/Documents/Study/Deep_Learning/FINAL/Image/hlp.jpg")
        hlp_img_btn=hlp_img_btn.resize((180,180),Image.ANTIALIAS)
        self.hlp_img1=ImageTk.PhotoImage(hlp_img_btn)

        hlp_b1 = Button(bg_img,command=self.helpSupport,image=self.hlp_img1,cursor="hand2",)
        hlp_b1.place(x=370,y=350,width=180,height=180)

        hlp_b1_1 = Button(bg_img,command=self.helpSupport,text="Help Support",cursor="hand2",font=("times new roman",15,"bold"),bg="darkblue",fg="white")
        hlp_b1_1.place(x=370,y=510,width=180,height=45)
        # Button 6: Exit
        exi_img_btn=Image.open(r"C:/Users/Son/Documents/Study/Deep_Learning/FINAL/Image/exi.jpg")
        exi_img_btn=exi_img_btn.resize((180,180),Image.ANTIALIAS)
        self.exi_img1=ImageTk.PhotoImage(exi_img_btn)

        exi_b1 = Button(bg_img,command=self.Close,image=self.exi_img1,cursor="hand2",)
        exi_b1.place(x=830,y=350,width=180,height=180)

        exi_b1_1 = Button(bg_img,command=self.Close,text="Exit",cursor="hand2",font=("times new roman",15,"bold"),bg="darkblue",fg="white")
        exi_b1_1.place(x=830,y=510,width=180,height=45)
        
# #==================Funtion for Open Images Folder==================
#     def open_img(self):
#         os.startfile("data_img")
# ==================Functions Buttons=====================
    def student_pannels(self):
        self.new_window=Toplevel(self.root)
        self.app=Student(self.new_window)
    def emotiondetector(self):
        self.app = EmotionDetector(self.root)
    def attendance(self):
        self.new_window=Toplevel(self.root)
        self.app=Attendance(self.new_window)
    def photos(self):
        self.new_window=Toplevel(self.root)
        self.app=Photos(self.new_window)    
    def helpSupport(self):
        self.new_window=Toplevel(self.root)
        self.app=Helpsupport(self.new_window)
    def Close(self):
        root.destroy()

if __name__== "__main__":
    root = Tk()
    obj=Main(root)
    root.mainloop()
