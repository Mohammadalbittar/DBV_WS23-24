import cv2
import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def extract_background(video_cap,sampling_interval):
    '''
    extracts Background from video and saves it as an image

    :param video_cap: video material
    :param sampling_interval: Sample to be taken after x frames
    :return: background_image: Background_image
    '''

    #intialize parameters
    cap = video_cap
    sampling_interval = 500

    num_frames = cap.get(cv.CAP_PROP_FRAME_COUNT)    #get number of frames


    #Sample frames
    frame_id = 0                                     #Start at frame 0
    frames_ids = []                                  #Sampled frames id
    frames = []                                      #Sampled frames

    while frame_id < num_frames:
        frames_ids.append(frame_id)
        frame_id += sampling_interval

    for i in frames_ids:
        cap.set(cv.CAP_PROP_POS_FRAMES,i)
        ret, frame = cap.read()
        frames.append(frame)

    background_image = np.median(frames, axis=0).astype(np.uint8)
    cv.imwrite("Background.jpg",background_image)
    return background_image

def find_Stats_point(video_cap,Background_image):
    '''

    :param video_cap:
    :param Background_image:
    :return:
    '''
    cap = video_cap
    bg_gray = cv.cvtColor(Background_image,cv.COLOR_BGR2GRAY)
    img = cv.GaussianBlur(bg_gray,(3,3),0)
    fg_roi = cv.createBackgroundSubtractorKNN(dist2Threshold=600)

    points_stat = []

    frame_counter = 34
    while True:
        print(frame_counter)
        cap.set(cv.CAP_PROP_POS_FRAMES,frame_counter)
        ret, frame = cap.read()
        #if not ret:
            #print("hier")
            #break

        fgmask = fg_roi.apply(frame)
        fgmask = cv.medianBlur(fgmask,7)

        frame_gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        frame_gray = cv.GaussianBlur(frame_gray,(3,3),0)

        diff = cv.absdiff(frame_gray,img)
        diff[diff < 90] = 0
        diff[diff > 0] = 255
        element = cv.getStructuringElement(cv.MORPH_DILATE,(7,7))
        diff = cv.dilate(diff,element)
        diff = cv.medianBlur(diff,7)

        static_mask = fgmask < 1
        static_frame = np.copy(diff)
        static_frame[~static_mask]=0

        countours, _ = cv.findContours(static_frame,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)


        for cnt in countours:
            area = cv.contourArea(cnt)
            if area > 800:
                x, y, w, h = cv.boundingRect(cnt)
                cx = (x+x+w)//2
                cy = (y+y+h)//2
                cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
                cv.circle(frame,(cx,cy),2,(0,0,255),-1)
                points_stat.append((cx,cy))

        cv.imshow("Original Frame",frame)
        cv.imshow("Foreground Mask", fgmask)
        cv.imshow("Diff mask", diff)
        cv.imshow("Not Moving",static_frame)
        frame_counter += 1

        key = cv.waitKey(30)
        if key == 27:
            break



    np.save("Points_Stationary",points_stat)
    cap.release()
    cv.destroyAllWindows()
    points = points_stat
    return points

def find_rois_points(background_image,Stat_points):
    img = background_image
    points = Stat_points
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(points)

    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    plt.scatter(points[:, 0], points[:, 1], c=labels, cmap='viridis', edgecolor='k', s=50)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Cluster Centers')
    plt.title('K-Means Clustering')
    plt.xlabel('X_Pos')
    plt.ylabel('Y_pos')
    plt.legend()
    plt.show()

    dimensions = np.shape(img)
    blank = np.full_like(img,fill_value=255)
    blank = cv2.cvtColor(blank,cv2.COLOR_BGR2GRAY)

    for pnt in points:
        cv2.circle(img, (pnt[0],pnt[1]),2,(0,0,255),-1)

    for cnt in centers:
        cv2.circle(img, (round(cnt[0]), round(cnt[1])), 9, (0, 255, 0), 3)

    rounded_centers = []
    sorted_indexs = np.argsort(centers[:,0])
    sorted_centers = centers[sorted_indexs]

    row_cnts = 0
    for cnt in sorted_centers:
        cnt_1 = round(cnt[0])
        cnt_2 = round(cnt[1])
        rounded_centers.append([cnt_1,cnt_2])
        if row_cnts == 1 or row_cnts == 2 :
            cv2.line(img, (0, round(cnt[1])), (dimensions[1], round(cnt[1])), (255, 0, 255), 3)
            cv2.line(blank, (0, round(cnt[1])), (dimensions[1], round(cnt[1])), (0, 0, 0), 1)
        else:
            cv2.line(img, (round(cnt[0]), 0), (round(cnt[0]), dimensions[0]), (255, 0, 0), 3)
            cv2.line(blank, (round(cnt[0]), 0), (round(cnt[0]), dimensions[0]), (0, 0, 0), 1)

        row_cnts+=1

    contours, _ = cv.findContours(blank,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    intersections = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        intersections.append((x, y))

    intersections = np.array(intersections)
    # remove zeros
    rows_to_delete = np.any(intersections == 0, axis=1)
    intersections = intersections[~rows_to_delete]

    for pts in intersections:
        cv2.circle(img,(pts[0],pts[1]),6,(255,255,0),-1)

    cv.imshow("Points",img)
    cv.imshow("Blank", blank)
    #cv.imshow("detected rectangles", img2)
    cv.waitKey(0)
    cv2.destroyAllWindows()
    return intersections