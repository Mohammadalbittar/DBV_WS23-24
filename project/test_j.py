import cv2

cap = cv2.VideoCapture('your_video.mp4')



def backsub(cap):
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

    while True:
        ret, frame = cap.read()
        if frame is None:
            break

        fgMask = backSub.apply(frame)

        cv2.imshow('Frame', frame)
        cv2.imshow('Foreground Mask', fgMask)

        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

