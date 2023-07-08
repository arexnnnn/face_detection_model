import cv2

capture = cv2.VideoCapture(0)  # here we can pass the video link or whatever
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
    ret, frame = capture.read()  # a return value and an image

    if ret:
        faces = classifier.detectMultiScale(frame)

        for face in faces:
            x, y, w, h = face
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)

        cv2.imshow("My Window", frame)  # if we have a return value then only we can show image in our window

    key = cv2.waitKey(20)   # occurrence, duration, process, frame rate

    if key == ord("q"):   # ord is a unicode value
        break

capture.release()
cv2.destroyAllWindows()
