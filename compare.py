from face_functions import (
    get_face_embedding,
    compare_faces
)


embedding_a = get_face_embedding("student_a.jpg")

embedding_c = get_face_embedding("student_c.jpg")


if embedding_a is None or embedding_c is None:
    print("Could not compare faces")
    exit()


same_face, similarity = compare_faces(
    embedding_a,
    embedding_c,
    threshold=0.5
)


print("Similarity:", similarity)


if same_face:
    print("SAME FACE")
else:
    print("DIFFERENT FACE")