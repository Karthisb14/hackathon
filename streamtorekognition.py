import boto3
import cv2

rekognition = boto3.client('rekognition', region_name='us-east-1')

cap = cv2.VideoCapture(0)
frame_count = 0
N = 30  # call Rekognition every 30 frames (~1 sec if FPS = 30)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Only send every Nth frame to Rekognition
    if frame_count % N == 0:
        small_frame = cv2.resize(frame, (640, 480))
        _, buffer = cv2.imencode('.jpg', small_frame)
        img_bytes = buffer.tobytes()

        # Rekognition call
        response = rekognition.detect_faces(
            Image={'Bytes': img_bytes},
            Attributes=['ALL']
        )
        print(response.get('FaceDetails', [['gender', 'age range', 'emotions']]))
        #print(response.get('FaceDetails', []))
        #print(response.get('FaceDetails', ['gender', 'age range', 'emotions']))

        # Draw boxes
        height, width, _ = small_frame.shape
        for faceDetail in response['FaceDetails']:
            box = faceDetail['BoundingBox']
            left = int(box['Left'] * width)
            top = int(box['Top'] * height)
            w = int(box['Width'] * width)
            h = int(box['Height'] * height)

            cv2.rectangle(small_frame, (left, top), (left + w, top + h), (0, 255, 0), 2)

            age_range = faceDetail['AgeRange']
            cv2.putText(small_frame, f"Age: {age_range['Low']}-{age_range['High']}",
                        (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Display original or annotated frame
    cv2.imshow("Webcam Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()