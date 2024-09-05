from flask import Flask, render_template, Response
import cv2
import face_recognition
import numpy as np
import time

app = Flask(__name__)

# Initialize camera and set resolution to reduce load
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
if not camera.isOpened():
    raise RuntimeError("Could not start camera.")

# Load and encode known faces
sakshi_image = face_recognition.load_image_file("Sakshi/sakshi.jpg")
sakshi_face_encoding = face_recognition.face_encodings(sakshi_image)[0]

Vipin_image = face_recognition.load_image_file("Vipin/Vipin.jpeg")
Vipin_face_encoding = face_recognition.face_encodings(Vipin_image)[0]

known_face_encodings = [sakshi_face_encoding, Vipin_face_encoding]
known_face_names = ["sakshi", "Vipin"]

# Global flag to process every alternate frame
process_this_frame = True

def gen_frames():
    global process_this_frame
    while True:
        success, frame = camera.read()
        if not success:
            print("Failed to grab frame")
            break

        # Process every alternate frame to reduce CPU usage
        if process_this_frame:
            # Resize frame for faster face recognition
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            
            # Ensure the frame is in RGB format as required by face_recognition
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Detect faces in the frame
            face_locations = face_recognition.face_locations(rgb_small_frame)

            # Debugging: Check if face locations are being detected
            print(f"Face locations detected: {face_locations}")

            # Extract face encodings from the detected face locations
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            face_names = []

            # Debugging: Check if face encodings are being generated
            print(f"Detected {len(face_encodings)} face encodings")

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # Use the known face with the smallest distance
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)
                print(f"Recognized face: {name}")

        process_this_frame = not process_this_frame  # Toggle frame processing

        # Draw boxes and labels around recognized faces
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            print(f"Drawing rectangle for {name} at ({left}, {top}, {right}, {bottom})")

            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Convert the frame to JPEG format and yield it
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        time.sleep(0.1)  # Add delay to control the frame rate


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
