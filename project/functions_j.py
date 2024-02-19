import cv2 as cv
import pafy
import pytube
from pytube import YouTube


def get_video(url:str):
    video = pafy.new(url)       #Create pafy object with youtube URL
    best_qual_stream =video.getbest()   # get the best quality of the stream
    cap = cv.VideoCapture(best_qual_stream.url)        # OpenCV Video Capture Objekt

    return cap      # Return named CV Video Capture Object

def get_livestream(url:str):        #Funktioniert nicht
    youtube_stream = YouTube(url)
    stream = youtube_stream.streams.filter(only_video=True).first()

    cap = cv.VideoCapture(stream.url)
    return cap

