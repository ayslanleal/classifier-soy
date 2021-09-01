import sys
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import msvcrt
from glob import glob
import tkinter
from tkinter import messagebox,filedialog
from datetime import datetime

def detection(image):
    image_in = image
   
    height, width = image_in.shape[:2]
    
    
    mask = np.zeros(image_in.shape[:2],np.uint8)
       
       
    bgdarray = np.zeros((1,65),np.float64)
    
    fgdarray = np.zeros((1,65),np.float64)

    rect_size = (20,20,width-30,height-30)
    
    cv2.grabCut(image_in,mask,rect_size,bgdarray,fgdarray,5,cv2.GC_INIT_WITH_RECT)
    
    mask_output = np.where((mask==2)|(mask==0),0,1).astype('uint8')
  
    image_seg = image_in*mask_output[:,:,np.newaxis]
   

    background_img = np.zeros((image.shape), dtype = "uint8")

    background_img[np.where((background_img > [0,0,0]).all(axis = 2))] =[255,255,255]
   
    image_segmented = background_img + image_seg
    
    image_hsv = cv2.cvtColor(image_segmented, cv2.COLOR_BGR2HSV)
    
    hsv_planes = cv2.split(image_hsv)
    brightness = hsv_planes[2]
    useEqualize = 1
    blur_Size = 21
    threshold_val = int(33.0 * 255 / 100)
    temp=brightness
    
    if (blur_Size >= 3):
        blur_Size += (1 - blur_Size % 2)
        temp = cv2.GaussianBlur(temp, (blur_Size, blur_Size), 0)
    if (useEqualize):
        temp = cv2.equalizeHist(temp)
        
    
    ret, temp = cv2.threshold(temp, threshold_val, 255, cv2.THRESH_BINARY_INV)
    
    
    contours, hierarchy = cv2.findContours(temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    image_processed = image1.copy()
    Contour_largest = 0
    largest = -1
    for i in range(len(contours)):
        contour_size = len(contours[i])
        if (contour_size > Contour_largest):
            Contour_largest = contour_size
            largest = i
   
    img_mask = np.zeros(image.shape, np.uint8)
  
    if (largest > 0):
        Final = contours[largest]
        rect = cv2.minAreaRect(Final)
        rect_box = cv2.boxPoints(rect)
        rect_box = np.int0(rect_box)
       
        image_result = cv2.drawContours(image_processed,[rect_box],0,(0,0,255),2)
       
    return image_processed
    
if __name__ == '__main__':  
    isError = 0
        
    parent = tkinter.Tk() 
    
    parent.overrideredirect(1)
    
    parent.withdraw()
    parent.attributes("-topmost", True)
    
    info = messagebox.showinfo('Information', 'Select a folder to proceed', parent=parent)
    directory = filedialog.askdirectory(title='Select Image Folder', parent=parent)

    ext = ['png', 'jpg'] 
    start_time = datetime.now()
    if not os.path.exists(directory+"/output"):
        os.makedirs(directory+"/output")
    
    image_names=[]
    file1 = open(directory+"/output/Output.txt","w+")
    
    for e in ext:
        image_names += glob(directory + '/**/*.' + e)
    
    file1.write("Found " + str(len(image_names)) + " images in the selected folder \n \n")
    
    if len(image_names)==0:
        isError = 1
        error = messagebox.showerror('FileNotFoundError', 'No images in the selected folder', parent=parent)
        exit()
    i=1
    
    for file in image_names:
        file1.write("Processing.. " + file+"\n")
        try:
            image = cv2.imread(file)
            image = cv2.resize(image,(227,227))
            image1 = image
            image2 = image
        except:
            isError = 1            
            error = messagebox.showerror('FileReadError', 'Error reading file', parent=parent)
            exit()
        output=detection(image)
        cv2.imwrite(directory+"/output/New_{:>01}.jpg".format(i),output)
        i=i+1
    end_time = datetime.now()
    compute_time=end_time - start_time
    hours, remainder = divmod(compute_time.total_seconds(),60*60)
    minutes, seconds = divmod(remainder,60)
    
    if (isError != 1):
        info = messagebox.showinfo('Processing Successful', "Find the Processed images in \n \n " +directory+"/output/", parent=parent)        
    file1.write("\n\n Execution time : {} hrs: {} mins : {} secs".format(int(hours),int(minutes),seconds))
    file1.close()