import cv2
import numpy as np
import os
from face_detect import FaceDetect
from face_recognize import FaceRecognition


def main():
    # onnx file can be downloaded from https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
    detector = FaceDetect(model_path='models/face_detection_yunet_2023mar_int8bq.onnx')

    # onnx file can be downloaded from https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface
    recognizer = FaceRecognition(model_path='models/face_recognition_sface_2021dec_int8bq.onnx')

    feature_data = dict()

    image_dir = os.path.join(os.path.dirname(__file__), "images")
    for (root, dirs, files) in os.walk(image_dir, topdown=True):
        for file in files:
            name = os.path.basename(root).lower().replace(" ", "_")
            image_file = os.path.join(root, file)
            img = cv2.imread(image_file)
            detector.set_input_size([img.shape[1], img.shape[0]])
            result, faces = detector.detect_face(img)

            if faces is not None:
                for face in faces:
                    feature = recognizer.get_feature(img, face[:-1])
                    feature_data.update({name: feature})

            else:
                feature_data.update({name: "Face not found"})

    cap = cv2.VideoCapture(0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Get the frame width
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Get the frame height
    
    tm = cv2.TickMeter()

    detector.set_input_size([w, h])
    '''
    For distance type I'm using FR_COSINE 
    Check the following reference check the following documentation and sample
    Docs: https://docs.opencv.org/4.x/da/d09/classcv_1_1FaceRecognizerSF.html#a6501674e36c7283491db853ed6599b58
    Sample: https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/sface.py 
    '''
    cosine_threshold = 0.363 # Use this if the distance type FR_COSINE (dist_type = 0)
    # norml2_threshold = 1.128 #  Use this if the distance type FR_NORM_L2 (dist_type = 1)

    while True:
        ret , frame = cap.read()
        frame = cv2.flip(frame, 1) # For better reconition we need to flip the frame first
        tm.start()
        idx, cam_faces = detector.detect_face(frame)
        tm.stop()

        if cam_faces is not None:
            for cam_face in cam_faces:
                cam_feature = recognizer.get_feature(frame, cam_face[:-1])
                name = "Not recognized"
                score = 0
                for (label, img_feature) in feature_data.items():

                    cosine_score = recognizer.get_match(cam_feature, img_feature, 0)
                    if cosine_score > cosine_threshold:
                        name = label
                        score = cosine_score
                        print("Name: ", label, "Score: ", score)
                coord = tuple(map(np.int32, cam_face[:-1]))

                cv2.rectangle(frame, (coord[0], coord[1]), (coord[0] + coord[2], coord[1] + coord[3]), (0, 255, 0), 2)
                cv2.putText(frame, "{}, Score: {:.2f}".format(name, score), (coord[0], coord[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()









    