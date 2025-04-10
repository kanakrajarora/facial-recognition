# 🎯 Face Recognition with OpenCV and face_recognition

This project implements **real-time face recognition** using a webcam feed. It detects faces, compares them with a known database, and labels them live on the video stream. It also saves the output video to disk.

---

## 📂 Project Structure

Face_recognition/
│
├── known_faces/           # Folder containing subfolders of known persons with images
│   ├── Alice/
│   │   └── img1.jpg
│   └── Bob/
│       └── img1.jpg
│
├── capture_and_save.py    # Script to generate face encodings from known images
├── recognize_from_video.py # Main script for face detection and recognition
├── face_database.npz      # Saved encodings and labels

---

## 🚀 How to Run

### 1. Install Dependencies

Make sure **Python 3.8–3.11** is installed. Then run:

```bash
pip install -r requirements.txt
```

### 2. Prepare Known Faces

Place images of people you want to recognize inside subfolders of the `known_faces/` directory. Each subfolder should be named after the person.

### 3. Generate Face Encodings

Run this script once to generate and save face encodings:

```bash
python capture_and_save.py
```

### 4. Run Face Recognition from Webcam
Start the recognition process:

```bash
python recognize_from_video.py
```
Press Q to quit the webcam window.
The video output will be saved as output.avi.

---

## Notes
1. Uses the hog model for face detection by default (faster on CPU).
2. You can switch to the cnn model (more accurate but slower) by changing the MODEL variable in the script.
3. Encodings are stored in a compressed .npz file.
