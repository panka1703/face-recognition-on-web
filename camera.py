import cv2
import face_recognition
import numpy as np
#cascPath = 'haarcascade_frontalface_dataset.xml'  # dataset
#faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)  # 0 for web camera live stream
#  for cctv camera'rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp'
#  example of cctv or rtsp: 'rtsp://mamun:123456@101.134.16.117:554/user=mamun_password=123456_channel=1_stream=0.sdp'
# Load a sample picture and learn how to recognize it.
Doan_image = face_recognition.load_image_file("Doan1.jpg")
Doan_face_encoding = face_recognition.face_encodings(Doan_image)[0]

biden_image = face_recognition.load_image_file("biden.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    Doan_face_encoding,
    biden_face_encoding
]
known_face_names = [
    "Huu Doan",
    "Joe Biden"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
#process_this_frame = True


def camera_stream():
    process_this_frame = True
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
         # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

    
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #faces = faceCascade.detectMultiScale(
         #   gray,
          #  scaleFactor=1.1,
           # minNeighbors=5,
            #minSize=(30, 30),
            #flags=cv2.CASCADE_SCALE_IMAGE
        #)

        # Draw a rectangle around the faces
        #for (x, y, w, h) in faces:
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the resulting frame in browser


        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                face_names.append(name)
        process_this_frame = not process_this_frame
        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right*=4
            bottom*=4
            left*=4
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        return cv2.imencode('.jpg', frame)[1].tobytes()
