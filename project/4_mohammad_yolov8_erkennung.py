from inference.models.utils import get_roboflow_model  # für prediction mit Roboflow
import supervision as sv
from ultralytics import YOLO
import cv2
# from super_gradients.training import models    # für prediction mit YOLO-nas

if __name__ == '__main__':

    source_video_path = 'resources/test_video_3.mp4'

    video_info = sv.VideoInfo.from_video_path(source_video_path)  # Informationen über das Video erhalten

    # Modell initialisieren
    model = get_roboflow_model('yolov8n-640')    # für prediction mit Roboflow
    # model = YOLO('yolov8n.pt')  # für prediction mit YOLO
    # model = model.get('yolo_nas_x', pretrained_weights='coco')    # für prediction mit YOLO-nas

    # Linienstärke dynamisch berechnen
    thickness = sv.calculate_dynamic_line_thickness(
        resolution_wh=video_info.resolution_wh
    )
    # Textskalierung dynamisch berechnen
    text_scale = sv.calculate_dynamic_text_scale(resolution_wh=video_info.resolution_wh)

    # Bounding-Box-Annotator initialisieren
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=thickness)
    # Label-Annotator initialisieren
    label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness)

    # Frame-Generator für das Video erstellen
    frame_generator = sv.get_video_frames_generator(source_video_path)

    for frame in frame_generator:
        # prediction des Modells erhalten
        result = model.infer(frame)[0]    # für prediction mit Roboflow
        # result = model(frame)[0]  # für prediction mit YOLO
        # result = model.predict(frame)[0]    # für prediction mit YOLO-nas

        # Detections aus den prediction extrahieren
        detections = sv.Detections.from_inference(result)    # für prediction mit Roboflow
        # detections = sv.Detections.from_ultralytics(result)  # für prediction mit YOLO
        # detections = sv.Detections.from_yolo_nas(result)    # für prediction mit YOLO-nas

        # Annotierte Frames erhalten
        annotated_frames = frame.copy()
        annotated_frames = bounding_box_annotator.annotate(
            scene=annotated_frames, detections=detections
        )
        annotated_frames = label_annotator.annotate(
            scene=annotated_frames, detections=detections
        )

        # Annotierte Frames anzeigen
        cv2.imshow('annotated Frames', annotated_frames)
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()
