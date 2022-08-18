import cv2

face_detector = cv2.CascadeClassifier('face_algorithm.xml')
smile_detector = cv2.CascadeClassifier('smile_algorithm.xml')

# grab webcam feed
webcam = cv2.VideoCapture(0)

# show current frame
while True:
    successful_frame_read, frame = webcam.read()
    if not successful_frame_read:
        break

    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face = face_detector.detectMultiScale(frame)

    for (x, y, w, h) in face:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 4)

        the_face = frame[y:y+h, x:x+h]

        # face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        smile = smile_detector.detectMultiScale(the_face, scaleFactor = 1.7, minNeighbors = 20)


        for (x_, y_, w_, h_) in smile:
            cv2.rectangle(the_face, (x_, y_), (x_+w_, y_+h_), (0, 255, 0), 2)


        if len(smile) > 0:
            cv2.putText(frame, 'ew', (x, y+h+40), fontScale = 2, fontFace = cv2.FONT_HERSHEY_PLAIN, color = (0, 255, 0), thickness = 2)

    cv2.imshow('Why so serious', frame)
    cv2.waitKey(1)

# clean up
webcam.release()
cv2.destroyAllWindows

cv2.waitKey()

print("Code Completed")

