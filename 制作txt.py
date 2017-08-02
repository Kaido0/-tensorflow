# -*- coding: utf-8 -*-

"""
制作类似于mnist.csv一样的数据集
最左列：label
784个像素点依次排在后
"""
import os
import scipy as scp
import scipy.misc
import matplotlib.pyplot as plt
from skimage import transform
import numpy as np
import pandas as pd
from PIL import Image

path = os.getcwd()

#先根据文件夹进行分类，改名
def pre():
    list=os.listdir(path)
    for file in list:
        count=1
        print file
        if 'changename.py' not in file:
            subpath=os.path.join(path,file)
            sublist=os.listdir(subpath)
            for subfile in sublist:

                oldname=os.path.join(subpath,subfile)
                newname=os.path.join(subpath,file+'_'+str(count)+'.jpg')
                count+=1
                # print oldname
                # print newname
                os.rename(oldname,newname)
        print "done"

def maketxt():
    list = os.listdir(path)
    file_w = open(path + "data.txt", 'w')
    count=0
    for file in list:
        count+=1
        if 'changename.py'  in file:
            continue
        if  '.idea' in file:
            continue
        subpath = os.path.join(path, file)
        sublist = os.listdir(subpath)
        for image_name in sublist:
            print image_name
            img1 = np.array(Image.open(os.path.join(subpath, image_name)).convert('L'))
            #img1 = scp.misc.imread(os.path.join(subpath,image_name))
            dst = transform.resize(img1, (28, 28))
            label=image_name.split('_')[0]
            label=np.resize(label,(1,1)).astype('float64')
            image=np.resize(dst,(1,784))
            l_img=np.hstack((label,image)) #label,image
            if count==1:
                data=np.array(l_img)
                count+=1
            else:
                data=np.vstack((data,l_img))
            # print data.shape
            # print data.dtype
            np.savetxt(path + "data.txt", l_img, fmt="%f")
            # print l_img.dtype
            # ima = np.resize(l_img[0, 1:], (28, 28))
            # plt.imshow(dst, cmap='gray')
            # plt.show()
    np.savetxt(path + "data.txt", data, fmt="%f")
    print "done"
    
#显示第c张图像
def readtxt():
    data = np.loadtxt(path + "data.txt")
    print data.shape
    print data.dtype
    c=1000
    print data[c,0]
    image=np.array(data[c,1:])
    print image.shape
    image=np.resize(image,(28,28))
    plt.imshow(image, cmap='gray')
    plt.show()

readtxt()
