# import required libraries
import cv2

# read the input image
img = cv2.imread('DSC_0638_1.jpg')

#resizing the image
scale_percent = 50 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

# Resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

# convert to grayscale of each frames
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# read the haarcascades to detect the faces in an image
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# detects faces in the input image
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
print('Number of detected faces:', len(faces))

# loop over all detected faces
if len(faces) > 0:
    for i, (x, y, w, h) in enumerate(faces):
        # To draw a rectangle in a face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        face = img[y:y + h, x:x + w]
        # Generate a unique filename for the cropped face image
        file_name = f'face{i}.jpg'
        cv2.imshow(f"Cropped Face {i}", face)
        cv2.imwrite("embedded_detectedPictures/"+file_name, face)
        print(f"face{i}.jpg is saved")
else:
    print("No faces detected")

# display the image with detected faces
cv2.imshow("image", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
