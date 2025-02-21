import cv2
import numpy as np
import os
import pickle
from face_detect import FaceDetect


class FaceRecognition :
    def __init__(self, model_path, config="", backend_id=0, target_id=0):
        self._model_path = model_path
        self._confg = config
        self._backend_id = backend_id
        self._target_id = target_id

        self._recognizer = cv2.FaceRecognizerSF.create(self._model_path, 
                                                       self._confg, 
                                                       self._backend_id, 
                                                       self._target_id)
    
    def get_feature(self, image, face):
        allign_face = self._recognizer.alignCrop(image, face)
        face_feature = self._recognizer.feature(allign_face)
        return face_feature
    
    def get_match(self, feature1, feature2, dist_type):
        match_score = self._recognizer.match(feature1, feature2, dist_type)
        return match_score


if __name__ == "__main__":
    detector = FaceDetect(model_path='models/face_detection_yunet_2023mar_int8bq.onnx')
    recognizer = FaceRecognition(model_path='models/face_recognition_sface_2021dec_int8bq.onnx')
    image_dir = os.path.join(os.path.dirname(__file__), "images")

    for (root, dirs, files) in os.walk(image_dir, topdown=True):
        print("root:", root)
        print("dirs: ", dirs)
        print("files: ", files)
        for file in files:
            print(os.path.basename(root))
            image_file = os.path.join(root,file)     
            img  = cv2.imread(image_file)        
            print("File: ", os.path.join(root,file) )            
            detector.set_input_size([img.shape[1], img.shape[0]])
            result, faces = detector.detect_face(img)            

            if faces is not None:
                for face in faces:
                    print("face: ", face[:-1])
                    feature = recognizer.get_feature(img, face[:-1])
                    print(feature)
            else:
                print("Face not found")
            

