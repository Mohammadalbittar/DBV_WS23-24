import cv2 as cv
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.solutions import speed_estimation
from collections import defaultdict


track_history = defaultdict(lambda: [])

model = YOLO("yolov8n-seg.pt")   # segmentation model
names = model.model.names

cap = cv.VideoCapture("../video/traffic.mp4")
w, h, fps = (int(cap.get(x)) for x in (cv.CAP_PROP_FRAME_WIDTH, cv.CAP_PROP_FRAME_HEIGHT, cv.CAP_PROP_FPS))

out = cv.VideoWriter('../video/instance-segmentation-object-tracking.avi', cv.VideoWriter_fourcc(*'MJPG'), fps, (w, h))


# Init speed-estimation obj
line_pts = [(0, 360), (1280, 360)]
#speed_obj = speed_estimation.SpeedEstimator()
#speed_obj.set_args(reg_pts=line_pts, names=names, view_img=True)


while True:
    ret, im0 = cap.read()
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    annotator = Annotator(im0, line_width=2)

    results = model.track(im0, persist=True, device="mps")

    #im1 = speed_obj.estimate_speed(im0, results)

    if results[0].boxes.id is not None and results[0].masks is not None:
        masks = results[0].masks.xy
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for mask, track_id in zip(masks, track_ids):
            annotator.seg_bbox(mask=mask,
                               mask_color=colors(track_id, True),
                               track_label=str(track_id))

    #out.write(im0)
    cv.imshow("instance-segmentation-object-tracking", im0)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cap.release()
cv.destroyAllWindows()