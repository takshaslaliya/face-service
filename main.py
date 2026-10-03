import cv2

face_detector = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

image = cv2.imread("test.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_detector.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5
)

print("Number of faces:", len(faces))

for (x, y, width, height) in faces:
    cv2.rectangle(
        image,
        (x, y),
        (x + width, y + height),
        (0, 255, 0),
        2
    )

cv2.imshow("Face Detection", image)

cv2.waitKey(0)
cv2.destroyAllWindows()