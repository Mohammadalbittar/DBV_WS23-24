from project.opencv_instance import *
from project.functions_j import *
import cv2 as cv



path = "user_results/Zwischenergbnisse/traffic.mp4"
url = 'https://www.youtube.com/watch?v=2X27I6BAJcI'
def main():
    #lukas_kanade(path)
    #backround_sub(path, mog2=True)

    classic = True

    cap = get_livestream(url)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    length  = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    _, frame_one = cap.read()

    # bck Objekt erstellen, damit die BackroundSub Funktion eine Hisotrie erstellen kann und in der while Schleife permanent neu initialisiert wird

    mog = background_sub(methode='mog')
    knn = background_sub(methode='knn')
    cnt = background_sub(methode='cnt')
    gmg = background_sub(methode='gmg')
    motion = motion_extraction(cv.GaussianBlur(frame_one, (5, 5), sigmaX=4, sigmaY=5))
    lukas = dense_optical_flow_outer(frame_one)
    yolo_region = yolo_region_count()
    yolo_pred = yolo_predict()
    yolo_tracker = yolo_track()

    k = 0
    while k<38:
        ret, frame = cap.read()
        if not ret:
            print('Stream loading error')
            break

        print(f'Frame {k}/{length}')

        if classic==True:
            #frame1,_ = mog(frame)
            #frame2, _ = knn(frame)
            #frame3, _ = cnt(frame)
            #frame4, _ = gmg(frame)
            #frame5 = motion(frame)
            #frame6 = watershed_segmentation(frame1)
            _,frame7,_, _ = lukas(frame)
            #frame9 = hdbscan_clustering(frame5)

            #frame1 = add_text_to_frame(frame1, 'MOG')
            #frame2 = add_text_to_frame(frame2, 'KNN')
            #frame3 = add_text_to_frame(frame3, 'CNT')
            #frame4 = add_text_to_frame(frame4, 'GMG')
            #frame5 = add_text_to_frame(frame5, 'Motion')
            #frame6 = add_text_to_frame(frame6, 'watershed')
            frame7 = add_text_to_frame(frame7, 'Lukas Kanade')

        else:
            frame, ins, out = yolo_region(frame)
            print(f'In: {ins} Out: {out}')
            #frame = yolo_pred(frame)
            #frame = yolo_tracker(frame)

        #frame = cv.dilate(frame8, None, iterations=2)
        #q_, frame = cv.threshold(frame, 0.9, 1, cv.THRESH_BINARY)




        #frame = video_tiling_mixed(frame1, frame2, frame3, frame4, width, height)
        #frame = stitch_frames(frame2, frame7, frame5, frame3)

        # Zeigen der Ergebnisse
        cv.imshow('Live_Output', frame7)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        str = "Lukas_Kanade_1"
        cv.imwrite(f'resources/{str}.png', frame7)
        k +=1


    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
