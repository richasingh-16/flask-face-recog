from flask import Flask, request, jsonify
import face_recognition
import cv2
import numpy as np
import os

app = Flask(__name__)

# Load known face encodings
known_face_encodings = []
known_face_names = []

known_dir = "known_faces"
for filename in os.listdir(known_dir):
    filepath = os.path.join(known_dir, filename)
    image = face_recognition.load_image_file(filepath)
    encodings = face_recognition.face_encodings(image)
    if encodings:
        known_face_encodings.append(encodings[0])
        known_face_names.append(os.path.splitext(filename)[0])

@app.route("/recognize", methods=["POST"])
def recognize():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    img_np = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_img)
    face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

    recognized = []

    for encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, encoding)
        name = "Unknown"
        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]
        recognized.append(name)

    return jsonify({"names": recognized})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
