from project.opencv_instance import *
from project.functions_j import *
import cv2 as cv

path = "video/traffic.mp4"
url = 'https://www.youtube.com/watch?v=Ellzen6Z7t8'
def main():
    #lukas_kanade(path)
    #backround_sub(path, mog2=True)

    cap = get_livestream(url)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    _, frame_one = cap.read()

    # bck Objekt erstellen, damit die BackroundSub Funktion eine Hisotrie erstellen kann und in der while Schleife permanent neu initialisiert wird

    mog = background_sub(methode='mog')
    knn = background_sub(methode='knn')
    cnt = background_sub(methode='cnt')
    gmg = background_sub(methode='gmg')
    motion = motion_extraction(cv.GaussianBlur(frame_one, (5, 5), sigmaX=4, sigmaY=5))

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Stream loading error')
            break




        frame1,_ = mog(frame)
        frame2, _ = knn(frame)
        frame3, _ = cnt(frame)
        frame4, _ = gmg(frame)
        frame5 = motion(frame)

        frame1 = add_text_to_frame(frame1, 'MOG')
        frame2 = add_text_to_frame(frame2, 'KNN')
        frame3 = add_text_to_frame(frame3, 'CNT')
        frame4 = add_text_to_frame(frame4, 'GMG')
        frame5 = add_text_to_frame(frame5, 'Motion')


        frame = video_tiling(frame5, frame2, frame3, frame4, width, height, Color=False)


        # Zeigen der Ergebnisse
        cv.imshow('Live_Output', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()