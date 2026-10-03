import cv2
from insightface.app import FaceAnalysis

# Create the face analysis model
app = FaceAnalysis(
    name="buffalo_l"
)

# Prepare the model
app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)

# Read the image
image = cv2.imread("test.jpg")

# Check if image was loaded
if image is None:
    print("Image could not be loaded")
    exit()

# Detect faces
faces = app.get(image)

# Print number of faces
print("Number of faces:", len(faces))

# Process every detected face
for face in faces:

    # Get face embedding
    embedding = face.embedding

    print("Embedding shape:", embedding.shape)

    print("First 10 values:")
    print(embedding[:10])