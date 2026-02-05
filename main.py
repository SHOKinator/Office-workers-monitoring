import cv2
import numpy as np

from yolo import model

video_path = "assets/office.mp4"
output_path = "output/final_rtdetr-x.mp4"

cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

wp1 = np.array([[250, 800], [150, 970], [420, 1000], [400, 800]])
wp2 = np.array([[750, 620], [820, 530], [930, 600], [820, 700]])
wp3 = np.array([[480, 390], [550, 380], [650, 470], [550, 520]])
wp4 = np.array([[1180, 350], [1220, 300], [1300, 320], [1270, 370]])
wp5 = np.array([[930, 280], [980, 260], [1050, 300], [1000, 350]])
zones = [wp1, wp2, wp3, wp4, wp5]

employees = {}

def format_time(seconds):
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hrs:02}:{mins:02}:{secs:02}"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, classes=[0], tracker="bytetrack.yaml", persist=True)
    annotated_frame = frame.copy()

    for zone in zones:
        pts = zone.reshape((-1, 1, 2))
        cv2.polylines(annotated_frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    if results[0].boxes.id is not None:
        for box, track_id_raw in zip(results[0].boxes, results[0].boxes.id):
            track_id = int(track_id_raw)

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            center_point = (int(center_x), int(center_y))

            is_in_zone = any(cv2.pointPolygonTest(zone, center_point, False) >= 0 for zone in zones)

            if track_id not in employees:
                employees[track_id] = {
                    "frames_in_zone": 0
                }

            if is_in_zone:
                employees[track_id]["frames_in_zone"] += 1

            seconds_in_zone = employees[track_id]["frames_in_zone"] / fps
            formatted_time = format_time(seconds_in_zone)

            box_color = (0, 255, 0) if is_in_zone else (0, 0, 255)
            status_text = "at the computer..." if is_in_zone else "walking..."

            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
            cv2.putText(annotated_frame, status_text, (int(x1), int(y1) - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 1)
            cv2.putText(annotated_frame, formatted_time, (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
            cv2.circle(annotated_frame, center_point, 3, box_color, -1)

    cv2.imshow('Employee Tracking', annotated_frame)
    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()