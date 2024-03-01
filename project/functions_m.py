import cv2
import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def extract_background(video_cap,sampling_interval_):
    '''
    extracts Background from video and saves it as an image

    :param video_cap: Video capture object
    :param sampling_interval: Sample to be taken after x frames
    :return: background_image: Background_image and number of used frames
    '''

    # Intialize parameters
    cap = video_cap
    sampling_interval = sampling_interval_

    # Get total number of frames
    num_frames = cap.get(cv.CAP_PROP_FRAME_COUNT)


    #Sample frames
    frame_id = 0          # Start at frame 0
    frames_ids = []       # Sampled frames id
    frames = []           # Sampled frames

    while frame_id < num_frames:
        frames_ids.append(frame_id)
        frame_id += sampling_interval

    # Read sampled frames
    for i in frames_ids:
        cap.set(cv.CAP_PROP_POS_FRAMES,i)
        ret, frame = cap.read()
        frames.append(frame)

    # Compute the median of sampled frames to obtain background image
    background_image = np.median(frames, axis=0).astype(np.uint8)
    # Save the background image as PNG file
    cv.imwrite("Background.png",background_image)

    # Get the number of used frames
    used_frames = len(frames)

    # Return background image and no. of used frames
    return background_image, used_frames

def find_Stats_point(video_cap,Background_image):
    '''
    Find stationary points in a video using background subtraction and contour detection

    :param video_cap: Video capture object
    :param Background_image: Background image for background subtraction
    :return:  List of stationary points (x,y - coordinates) and number of used frame
    '''

    # Intialize video capture, convert the background image to grayscale, and apply
    cap = video_cap
    bg_gray = cv.cvtColor(Background_image,cv.COLOR_BGR2GRAY)
    img = cv.GaussianBlur(bg_gray,(3,3),0)

    # Create a KNN background subtractor
    fg_roi = cv.createBackgroundSubtractorKNN(dist2Threshold=600)

    # List to store stationary points
    points_stat = []

    # Get total number of frames in the video
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    used_frames = 1
    # Loop through each frame
    for frame_number in range(frame_count):
        # Read the current frame
        ret, frame = cap.read()

        # skip frame if not read successfully
        if not ret:
            continue

        # Apply background subtraction(KNN) and median blur to the current frame
        fgmask = fg_roi.apply(frame)
        fgmask = cv.medianBlur(fgmask,7)

        # Convert the current frame to grayscale and apply Gaussian blur
        frame_gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        frame_gray = cv.GaussianBlur(frame_gray,(3,3),0)

        # Compute the abs difference between current frame and background image
        # to get 2nd mask
        diff = cv.absdiff(frame_gray,img)
        # Threshold the difference image
        diff[diff < 90] = 0
        diff[diff > 0] = 255
        # Morphological operations (diltaion) on the difference image
        element = cv.getStructuringElement(cv.MORPH_DILATE,(7,7))
        diff = cv.dilate(diff,element)
        diff = cv.medianBlur(diff,7)  #2nd Mask

        # Create a static objects mask by removing moving objects from differnce image
        static_mask = fgmask < 1
        static_frame = np.copy(diff)
        static_frame[~static_mask]=0

        # Find contours in the resulting static frame
        countours, _ = cv.findContours(static_frame,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

        # Iterate through the contours
        for cnt in countours:
            # Calculate the area of each contour, and use a threshold to filter out small objects
            area = cv.contourArea(cnt)
            if area > 800:
                x, y, w, h = cv.boundingRect(cnt)
                cx = (x+x+w)//2
                cy = (y+y+h)//2

                # Draw rectangles around stationary point
                cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
                # Draw a circle at the centroid of the stationary objects
                cv.circle(frame,(cx,cy),2,(0,0,255),-1)

                # Append the coordinates of the stationary point to the list
                points_stat.append((cx,cy))

        # Display frames for visualization
        cv.imshow("Original Frame",frame)
        cv.imshow("Foreground Mask", fgmask)
        cv.imshow("Diff mask", diff)
        cv.imshow("Not Moving",static_frame)

        #update number of used frames
        used_frames += 1

        # Break the loop if 'Esc' key is pressed
        key = cv.waitKey(30)
        if key == 27:
            break

    # Save the coordinates of stationary points to a file
    np.save("Points_Stationary",points_stat)

    # Release video capture and close OpenCV windows
    cap.release()
    cv.destroyAllWindows()

    # Return the list of stationary points
    points = points_stat
    return points, used_frames

def find_rois_points(background_image,Stat_points):
    '''
    Find region of interest (ROI) using stationary points und clustering methods (K-Means)

    :param background_image: Background image for processing
    :param Stat_points: List of stationary points (coordinates)
    :return: 3 points corners of ROI
    '''

    # Extract background image and stationary points
    img = background_image
    points = Stat_points

    # Apply K-Means clustering to stationary points
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(points)

    # Get cluster labels and centers
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # Clustering visualization
    #plt.scatter(points[:, 0], points[:, 1], c=labels, cmap='viridis', edgecolor='k', s=50)
    #plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Cluster Centers')
    #plt.title('K-Means Clustering')
    #plt.xlabel('X_Pos')
    #plt.ylabel('Y_pos')
    #plt.legend()
    #plt.show()

    # Creat a blank image
    dimensions = np.shape(img)
    blank = np.full_like(img,fill_value=255)
    blank = cv2.cvtColor(blank,cv2.COLOR_BGR2GRAY)

    # Draw stationary points on background image
    for pnt in points:
        cv2.circle(img, (pnt[0],pnt[1]),2,(0,0,255),-1)

    # Draw cluster centers on background image
    for cnt in centers:
        cv2.circle(img, (round(cnt[0]), round(cnt[1])), 9, (0, 255, 0), 3)

    # Sort cluster centers based on x position
    rounded_centers = []
    sorted_indexs = np.argsort(centers[:,0])
    sorted_centers = centers[sorted_indexs]

    # Draw vertical and horizontal lines from each center point to connect clusters and create rectangles
    # lines are drawn on background image for visualization and in Blank image to find lines intersection points
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

        row_cnts += 1

    # Find contours in the blank image to detect intersections
    contours, _ = cv.findContours(blank,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    intersections = []

    # Iterate through contours to get bounding rectangles
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        intersections.append((x, y))

    # Remove zeros from intersections array
    intersections = np.array(intersections)
    rows_to_delete = np.any(intersections == 0, axis=1)
    intersections = intersections[~rows_to_delete]

    # Draw circle on the orignal image for detected intersections
    for pts in intersections:
        cv2.circle(img,(pts[0],pts[1]),6,(255,255,0),-1)


    # Display images for viualization
    cv.imshow("Points",img)
    cv.imshow("Blank", blank)
    #cv.imshow("detected rectangles", img2)
    cv.waitKey(0)
    cv2.destroyAllWindows()

    # intersection points = corner of ROI
    roi_corners = intersections
    return roi_corners