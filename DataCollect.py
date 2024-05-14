"""
GROUP 07: 
Member : 
 1/ Dang Thanh Tuyen - 20110412
 2/ Danh Truong Son - 20110394
 3/ Dang Phuoc Truong Tai - 20110396
"""
import cv2
from tkinter import *

class datacollet():
        def __init__(self,root):
                self.root=root
                self.root.geometry("500x300")
                self.root.title("Please Enter Your Full Name")
                
                Label1 = Label(self.root,text="Enter Your Name: ",fg="black",font=("times new roman",15))
                Label1.grid(row=0,column=0,padx=5,pady=10)
                

                data = StringVar()
                textbox1 = Entry(self.root,textvariable=data,fg="black",font=("times new roman",15))
                textbox1.grid(row=0,column=1)

                def datacollect():
                        name='./AttendanceImage/'+ data.get() +'.jpg'
                        vid = cv2.VideoCapture(0)
                        while(True):
                                count = 0
                                ret,image = vid.read()
                                cv2.imshow("capture",image)
                                if cv2.waitKey(1) & 0xFF == ord("q"):
                                        break
                                cv2.imwrite(name, image)
                                count += 1
                                print("Capture Complete!!!")
                                if count > 0:
                                        break
                        vid.release()
                        cv2.destroyAllWindows()
                        
                button1 = Button(self.root,command=datacollect,text="Save",fg="black",font=("times new roman",15))
                button1.grid(row=1,column=1,sticky=W)

                Label2 = Label(self.root,fg="black",font=("times new roman",15))
                Label2.grid(row=2,column=1,sticky=W)

if __name__ == "__main__":
    root=Tk()
    obj=datacollet(root)
    root.mainloop()



    