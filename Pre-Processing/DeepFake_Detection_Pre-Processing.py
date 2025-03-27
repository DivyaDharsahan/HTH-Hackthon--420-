import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
import cv2 as cv
import glob
from IPython.display import HTML
from base64 import b64encode
import face_recognition
from PIL import Image

train_sample_metadata.info()
train_sample_metadata['label'].value_counts()
train_sample_metadata.groupby('label')['label'].count().plot(figsize=(10, 5), kind='bar', title='Distribution of Labels in the Training Set')
plt.show()

def unique_values(data):
    total = data.count()
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    uniques = []
    for col in data.columns:
        unique = data[col].nunique()
        uniques.append(unique)
    tt['Uniques'] = uniques
    return(np.transpose(tt))

unique_values(train_sample_metadata)

def most_frequent_values(data):
    total = data.count()
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    items = []
    vals = []
    for col in data.columns:
        itm = data[col].value_counts().index[0]
        val = data[col].value_counts().values[0]
        items.append(itm)
        vals.append(val)
    tt['Most frequent item'] = items
    tt['Frequency'] = vals
    tt['Percent from total'] = np.round(vals / total * 100, 3)
    return(np.transpose(tt))

most_frequent_values(train_sample_metadata)

original_counts = pd.DataFrame(train_sample_metadata['original'].value_counts())
original_counts.head(10)

fake_train_sample_video = list(train_sample_metadata.loc[train_sample_metadata.label=='FAKE'].sample(10).index)
fake_train_sample_video

real_train_sample_video = list(train_sample_metadata.loc[train_sample_metadata.label=='REAL'].sample(10).index)    # returning the index value which is video name
real_train_sample_video

def display_image_from_video(video_path):
    '''
    input: video_path - path for video
    process:
    1. perform a video capture from the video
    2. read the image
    3. display the image
    '''
    capture_image = cv.VideoCapture(video_path) 
    ret, frame = capture_image.read()
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)   # converting the frame color to RGB
    ax.imshow(frame)
    for video in fake_train_sample_video:
        display_image_from_video(os.path.join("/content/drive/MyDrive/Dataset/train_sample_videos/"+video))
    for video in real_train_sample_video:
        display_image_from_video(os.path.join("/content/drive/MyDrive/Dataset/train_sample_videos/"+video))



def play_video(video_file):
    '''
    Display video
    param: video_file - the name of the video file to display
    param: subset - the folder where the video file is located (can be TRAIN_SAMPLE_FOLDER or TEST_Folder)
    '''
    video_url = open(video_file,'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(video_url).decode()
    return HTML("""<video width=500 controls><source src="%s" type="video/mp4"></video>""" % data_url)
    play_video("/content/drive/MyDrive/Dataset/train_sample_videos/aapnvogymq.mp4")
    videos = glob.glob('/content/drive/MyDrive/Dataset/train_sample_videos/*.mp4')
    frame_cnt = []
    for video in videos:
        capture = cv.VideoCapture(video)
        frame_cnt.append(int(capture.get(cv.CAP_PROP_FRAME_COUNT)))
        print("Frames: ",frame_cnt)
        print("Avg Frame per video: ",np.mean(frame_cnt))

def image_from_video(video_path):
    '''
    input: video_path - path for video
    process:
    1. perform a video capture from the video
    2. read the image
    3. display the image
    '''
    capture_image = cv.VideoCapture(video_path) 
    ret, frame = capture_image.read()
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)   # converting the frame color to RGB
    ax.imshow(frame)

    return frame
    image = image_from_video("/content/drive/MyDrive/Dataset/train_sample_videos/aagfhgtpmv.mp4")
    face_locations = face_recognition.face_locations(image)
    
for face_location in face_locations:

        top,right,bottom,left = face_location
        print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
  
        face_image = image[top:bottom, left:right]
        fig, ax = plt.subplots(1,1, figsize=(5, 5))
        plt.grid(False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.imshow(face_image)