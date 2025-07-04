# !usr/bin/env python
# encoding:utf-8
from __future__ import division


'''
功能： 图像的数据预处理、标准化部分
'''


import os
import cv2
import time
import json


CONFIG_FILE = 'config.json'

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {}

current_config = load_config()
IMAGE_SIZE = current_config.get('image_size', 128)


def read_name_list(path):
    name_list = []
    for child_dir in os.listdir(path):
        name_list.append(child_dir)
    return name_list


def readAllImg(path,*suffix):
    '''
    基于后缀读取文件
    '''
    try:
        s=os.listdir(path)
        resultArray = []
        fileName = os.path.basename(path)
        resultArray.append(fileName)
        for i in s:
            if endwith(i, suffix):
                document = os.path.join(path, i)
                img = cv2.imread(document)
                resultArray.append(img)
    except IOError:
        # print("Error") # Commented out
        pass # Suppress error for now

    else:
        return resultArray


def endwith(s,*endstring):
    '''
    对字符串的后续和标签进行匹配
    '''
    resultArray = map(s.endswith,endstring)
    if True in resultArray:
        return True
    else:
        return False


def readPicSaveFace(sourcePath,objectPath,*suffix):
    '''
    图片标准化与存储
    '''
    # print(f"Processing images from {sourcePath} to {objectPath}") # Commented out
    if not os.path.exists(objectPath):
        os.makedirs(objectPath)
    try:
        resultArray=readAllImg(sourcePath,*suffix)
        count=1
        face_cascade=cv2.CascadeClassifier('config/haarcascade_frontalface_alt.xml')
        for i in resultArray:
            if type(i)!=str:
              gray=cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
              faces=face_cascade.detectMultiScale(gray, 1.3, 5)
              for (x,y,w,h) in faces:
                listStr=[str(int(time.time())),str(count)]  
                fileName=''.join(listStr)
                f=cv2.resize(gray[y:(y+h),x:(x+w)],(IMAGE_SIZE, IMAGE_SIZE))
                cv2.imwrite(objectPath+os.sep+'%s.jpg' % fileName, f)
                count+=1
    except Exception as e:
        # print("Exception: ",e) # Commented out
        pass # Suppress error for now
    else:
        person_name = os.path.basename(sourcePath.strip('/')).replace('/', '')
        # print(f'Successfully processed {count-1} faces for {person_name} to {objectPath}') # Commented out

if __name__ == '__main__':
    # print('dataProcessing!!!') # Commented out
    data_root_dir = 'data/'
    dataset_root_dir = 'dataset/'
    
    # Define common image suffixes
    image_suffixes = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.tiff', '.TIFF')

    # Iterate through all subdirectories in the data_root_dir
    for person_name in os.listdir(data_root_dir):
        source_path = os.path.join(data_root_dir, person_name)
        object_path = os.path.join(dataset_root_dir, person_name)
        
        # Check if it's a directory before processing
        if os.path.isdir(source_path):
            if os.path.exists(object_path):
                # print(f"Skipping {person_name}: Destination directory {object_path} already exists.") # Commented out
                continue
            readPicSaveFace(source_path, object_path, *image_suffixes)
