import face_recognition
import cv2
import numpy as np
import os

KNOWN_FACES_DIR = 'known_faces'
known_encodings = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    person_dir = os.path.join(KNOWN_FACES_DIR, name)
    if not os.path.isdir(person_dir):
        continue

    for filename in os.listdir(person_dir):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        path = os.path.join(person_dir, filename)
        print(f"[INFO] Processing {path}")
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(name)
            print(f"[SUCCESS] Encoding added for {filename}")
        else:
            print(f"[WARN] No face detected in {filename}")

np.savez_compressed("face_database.npz", encodings=known_encodings, names=known_names)
print("[DONE] Face encodings saved to 'face_database.npz'")