import face_recognition
import cv2
import numpy as np
import time

# Load database
data = np.load("face_database.npz", allow_pickle=True)
known_encodings = list(data['encodings'])
known_names = list(data['names'])

TOLERANCE = 0.6
MODEL = 'hog'
FRAME_THICKNESS = 2
FONT_THICKNESS = 1

def name_to_color(name):
    return [(ord(c.lower()) - 97) * 8 for c in name[:3]]

video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (frame_width, frame_height))

prev_time = 0

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Ensure frame is uint8 RGB (some cameras output different formats)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8)

    try:
        locations = face_recognition.face_locations(rgb_frame, model=MODEL)

        # Avoid error if no faces detected
        if not locations:
            cv2.imshow("Webcam", frame)
            if cv2.waitKey(1) == ord('q'):
                break
            continue

        encodings = face_recognition.face_encodings(rgb_frame, locations)
    except Exception as e:
        print(f"Error during face detection/encoding: {e}")
        continue

    for face_encoding, face_location in zip(encodings, locations):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, TOLERANCE)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            match = known_names[best_match_index]
            color = name_to_color(match)
        else:
            match = "Unknown"
            color = (0, 0, 255)

        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), color, FRAME_THICKNESS)
        cv2.rectangle(frame, (left, bottom), (right, bottom + 20), color, cv2.FILLED)
        label = f"{match}"
        if match != "Unknown":
            label += f" ({face_distances[best_match_index]:.2f})"
        cv2.putText(frame, label, (left + 5, bottom + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), FONT_THICKNESS)

    # FPS display
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
out.release()
cv2.destroyAllWindows()