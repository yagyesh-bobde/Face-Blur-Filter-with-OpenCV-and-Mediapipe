import cv2
import mediapipe as mp

vid = cv2.VideoCapture("./videos/video1.mp4") # Video Object
# Video by fauxels: https://www.pexels.com/video/close-up-video-of-man-wearing-red-hoodie-3249935
# 5 Video by Pixabay: https://www.pexels.com/video/tourist-crossing-the-street-855565/
faceDetectionClass = mp.solutions.face_detection # instanciate the face detection class
faceDetectionModel = faceDetectionClass.FaceDetection(min_detection_confidence=0.75) # instanciate the face detection model with confidence value

# Drawing Utils from mediapipe
mpDraw = mp.solutions.drawing_utils

while (True):

    ret, original = vid.read()  # Read Frame from Input -> returns (dimensions , frame)

    # replay the video
    if(not ret) :
        print("Replay video")
        vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue


    img = original.copy()

    HEIGHT, WIDTH , CHANNELS = img.shape

    #detect face
    # pass the image to faceDetectionModel to detect faces if any
    faces = faceDetectionModel.process(img)

    # loop over the face detections
    if(not faces.detections) :
        break;
    for faceDetection in faces.detections:
        # draw the bounding box of the face
        bbox = faceDetection.location_data.relative_bounding_box
        x1 , y1 , w, h = int(bbox.xmin*WIDTH) , int(bbox.ymin*HEIGHT) , int(bbox.width * WIDTH) , int(bbox.height * HEIGHT)

        # Crop the Face image
        crop_face = img[y1:y1+h, x1:x1+w]

        # Blur the cropped image
        blur_image = cv2.blur(img , (100 , 100))
        # apply the blur on original image
        blur_image[y1:y1+h, x1:x1+w] = crop_face

        # Draw the face bounding box
        mpDraw.draw_detection(img , faceDetection ,
                              keypoint_drawing_spec=mpDraw.DrawingSpec(thickness=0) ,
                              bbox_drawing_spec=mpDraw.DrawingSpec(color=(200,0,00) , thickness=10) )


    originalResize = cv2.resize(original, (720, 480))
    resizedFrame = cv2.resize(blur_image, (720, 480))

    cv2.imshow('Original Video', originalResize)
    cv2.imshow("Face Blur Filter", resizedFrame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()