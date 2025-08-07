import cv2
import boto3

# AWS Rekognition setup
rekognition = boto3.client('rekognition', region_name='us-east-1')

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")  # Or yolov4-tiny.weights + yolov4-tiny.cfg
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
classes = open("coco.names").read().strip().split('\n')

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    # Create blob and perform forward pass
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    detected_person = False

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = int(scores.argmax())
            confidence = scores[class_id]

            if confidence > 0.5:
                label = classes[class_id]
                if label == "person":
                    detected_person = True
                    print("🧍‍♂️ Person detected by YOLO!")
                    break
        if detected_person:
            break

    # If person is detected, send to Rekognition
    if detected_person:
        ret, jpeg = cv2.imencode('.jpg', frame)
        if ret:
            response = rekognition.detect_faces(
                Image={'Bytes': jpeg.tobytes()},
                Attributes=['ALL']
            )
            print("AWS Rekognition → Face details:", response['FaceDetails'])

    cv2.imshow("YOLO + Rekognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
