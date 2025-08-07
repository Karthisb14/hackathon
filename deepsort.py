import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import sys
import cv2
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
import boto3

# Initialize AWS Rekognition
try:
    rekognition = boto3.client("rekognition", region_name="us-east-1")
except Exception as e:
    print(f"Failed to initialize AWS Rekognition: {e}")
    sys.exit(1)

# Load YOLOv5 model with error handling
try:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.classes = [0]  # Only detect 'person'
except Exception as e:
    print(f"Failed to load YOLOv5 model: {e}")
    sys.exit(1)

# Initialize DeepSORT
try:
    tracker = DeepSort(max_age=30)
except Exception as e:
    print(f"Failed to initialize DeepSORT tracker: {e}")
    sys.exit(1)

# Open webcam
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")
except Exception as e:
    print(f"Webcam error: {e}")
    sys.exit(1)

# Keep track of already recognized people by track_id
recognized_track_ids = set()

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect with YOLO
        try:
            results = model(frame)
            detections = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, class]
        except Exception as e:
            print(f"Detection error: {e}")
            continue

        # Convert detections to format for DeepSORT
        det_list = []
        for *xyxy, conf, cls in detections:
            x1, y1, x2, y2 = map(int, xyxy)
            det_list.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

        # Track with DeepSORT
        try:
            tracks = tracker.update_tracks(det_list, frame=frame)
        except Exception as e:
            print(f"Tracking error: {e}")
            continue

        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"ID: {track_id}"

            # Run Rekognition only if not already done
            if track_id not in recognized_track_ids:
                person_crop = frame[y1:y2, x1:x2]
                if person_crop.size == 0:
                    print(f"Empty crop for ID {track_id}")
                    continue

                try:
                    _, img_bytes = cv2.imencode('.jpg', person_crop)
                    image_bytes = img_bytes.tobytes()

                    try:
                        response = rekognition.detect_faces(
                            Image={'Bytes': image_bytes},
                            Attributes=['ALL']
                        )

                        if response['FaceDetails']:
                            face = response['FaceDetails'][0]
                            gender = face.get('Gender', {}).get('Value', 'Unknown')
                            emotions = face.get('Emotions', [])
                            top_emotion = max(emotions, key=lambda e: e['Confidence']) if emotions else None
                            emotion_text = f"{top_emotion['Type']} ({top_emotion['Confidence']:.1f}%)" if top_emotion else "N/A"
                            
                            # Check for glasses
                            glasses = face.get('Eyeglasses', {}).get('Value', False)
                            sunglasses = face.get('Sunglasses', {}).get('Value', False)
                            glasses_text = ""
                            
                            if glasses:
                                glasses_text = " | Wearing Glasses"
                            elif sunglasses:
                                glasses_text = " | Wearing Sunglasses"
                            else:
                                glasses_text = " | No Glasses"

                            print(f"\nTrack ID: {track_id}")
                            print(f"Gender: {gender}")
                            print(f"Top Emotion: {emotion_text}")
                            print(f"Glasses: {'Yes' if glasses or sunglasses else 'No'}")

                            label += f" | {gender} | {emotion_text}{glasses_text}"
                        else:
                            print(f"No face detected for ID {track_id}")

                    except Exception as e:
                        print(f"Rekognition error for ID {track_id}: {e}")

                    recognized_track_ids.add(track_id)
                except Exception as e:
                    print(f"Image processing error for ID {track_id}: {e}")
            else:
                label += " | Already Identified"

            # Draw label
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("YOLOv5 + DeepSORT + Rekognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Program interrupted by user")
finally:
    cap.release()
    cv2.destroyAllWindows()