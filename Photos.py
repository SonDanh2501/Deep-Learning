"""
GROUP 10: 
Member : 
 1/ Danh Truong Son - 20110394
 2/ Nguyen Duc Huy - 20145449
 3/ Nguyen Trung Nguyen - 20110388
"""
from tkinter import *
from PIL import ImageTk, Image
from sys import path
import os

counter = 0

class Photos:
    def __init__(self,root):
        self.root=root
        # set up the tkinter window
        self.root.title("Group 10 - Member: Danh Truong Son - 20110394 / Nguyen Trung Nguyen - 20110388 / Nguyen Duc Huy - 20145449")
        self.root.geometry("700x500+0+0")
        # set up the images
        path = 'AttendanceImage'
        image_list=[]
        classNames = []
        myList = os.listdir(path)
        for images in myList:
            image = ImageTk.PhotoImage(Image.open("AttendanceImage/"+images).resize((600, 350)))
            image_list.append(image)
            classNames.append(os.path.splitext(images)[0]) # ==> Get The Name (Without the file type)

        # counter integer
        # change image function
        def ChangeImage():
            global counter
            if counter < len(image_list) - 1:
                counter += 1
            else:
                counter = 0
            imageLabel.config(image=image_list[counter])
            infoLabel.config(text=classNames[(counter)]  )
        # set up the components
        imageLabel = Label(root, image=image_list[0])
        infoLabel = Label(root, text=classNames[0], font="Helvetica, 20")
        button = Button(root, text="Change", width=20, height=2, bg="navyblue", font=("verdana",15,"bold"), command=ChangeImage)
        # display the components
        imageLabel.pack()
        infoLabel.pack()
        button.pack(side="bottom", pady=3)
        # run the main loop

if __name__== "__main__":
    root = Tk()
    obj=Photos(root)
    root.mainloop()
       
       