import cv2 as cv
from random import randrange

# trained data
trained_face_data = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml")

# choose an image to detect face
#img = cv.imread('RDJ.png')
# capture a video
webcam = cv.VideoCapture(0)

# Iterate forever over frames
while True:
    # read current frame
    successful_frame_read, frame = webcam.read()
    # convert to grayscale
    grayscaled_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    # draw a rectangle on the face
    for (x, y, w, h) in face_coordinates:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 10)
    cv.imshow('face detector', frame)
    key = cv.waitKey(1)
    # break out if Q key is pressed
    if key == 81 or key == 113:
        break

# release the VideoCapture object
webcam.release()

print("Code completed")
"""
# detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
print(face_coordinates)

# draw a rectangle on the face
for (x, y, w, h) in face_coordinates:
    cv.rectangle(img, (x, y), (x+w, y+h), (randrange(256),
                                           randrange(256), randrange(256)), 10)



#cv.imshow('image', img)
"""
