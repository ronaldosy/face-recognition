import cv2
import numpy as np
import os
import pickle
import argparse
from face_detect import FaceDetect
from face_recognize import FaceRecognition

'''
ONNX file can be downloaded from
detector: onnx file can be downloaded from https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
recognizer : onnx file can be downloaded from https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface
'''

# Get feature data from reference image, and stored into a picle file
def save_ref_feature():
    detector = FaceDetect(model_path='models/face_detection_yunet_2023mar_int8bq.onnx')
    recognizer = FaceRecognition(model_path='models/face_recognition_sface_2021dec_int8bq.onnx')
    feature_data = dict()        
    for (root, dirs, files) in os.walk('images/', topdown=True):
        print(root)
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
                feature_data.update({name: "No face found"})
                print("No face found for", name)
    
    data_file = open('data/feature_data', 'wb')
    pickle.dump(feature_data, data_file)
    data_file.close()


def main(cam_id:int = 0):    
    cap = cv2.VideoCapture(cam_id)

    if cap.isOpened(): # Check if the camera available or not
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Get the frame width
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Get the frame height
        detector = FaceDetect(model_path='models/face_detection_yunet_2023mar_int8bq.onnx')    
        recognizer = FaceRecognition(model_path='models/face_recognition_sface_2021dec_int8bq.onnx')

        # Read feature data from pickle file and load into dict
        feature_data = dict()
        try:
            with open('data/feature_data','rb') as data_file:
                feature_data = pickle.load(data_file)
                data_file.close()            
        except:
            feature_data.update({0:0})        

        detector.set_input_size([w, h])

        '''
        Check the following documentation and sample for distance type:
        Docs: https://docs.opencv.org/4.x/da/d09/classcv_1_1FaceRecognizerSF.html#a6501674e36c7283491db853ed6599b58
        Sample: https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/sface.py 
        '''
        cosine_threshold = 0.363 # Use this if the distance type FR_COSINE (dist_type = 0)
        # norml2_threshold = 1.128 #  Use this if the distance type FR_NORM_L2 (dist_type = 1)

        while True:
            ret , frame = cap.read()
            frame = cv2.flip(frame, 1) # For better reconition we need to flip the frame first

            _, cam_faces = detector.detect_face(frame)            
            
            if cam_faces is not None:
                for cam_face in cam_faces:
                    cam_feature = recognizer.get_feature(frame, cam_face[:-1])
                    coord = tuple(map(np.int32, cam_face[:-1]))
                    for (label, img_feature) in feature_data.items():                                        
                        if label == 0:
                            cv2.rectangle(frame, (coord[0], coord[1]), (coord[0] + coord[2], coord[1] + coord[3]), (0,0,255), 2)
                            cv2.putText(frame,"No data loaded", (coord[0], coord[1] - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)                                                    
                        else:
                            cosine_score = recognizer.get_match(cam_feature, img_feature, 0) # Use FR_COSINE as distance type

                            if cosine_score > cosine_threshold:                                                
                                cv2.rectangle(frame, (coord[0], coord[1]), (coord[0] + coord[2], coord[1] + coord[3]), (255,0,0), 2)
                                cv2.putText(frame, "{}".format(label.capitalize()), (coord[0], coord[1] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
                                break
                            else:                        
                                cv2.rectangle(frame, (coord[0], coord[1]), (coord[0] + coord[2], coord[1] + coord[3]), (0,0,255), 2)
                                cv2.putText(frame,"Not recognized", (coord[0], coord[1] - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)                                                                                                                                                
            else:
                cv2.putText(frame, "No face detected" , (5,20),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Face Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Camera not found")     

    
if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-g', '--generate-feature', action=argparse.BooleanOptionalAction, help="Generate feature data and store it") 
    arg_parser.add_argument('-c', '--camera-id',  help="Camera number in computer ")
    args = arg_parser.parse_args()

    if args.camera_id is not None:
        cam_id = int(args.camera_id)
    else:
        cam_id = 0

    if args.generate_feature:
        save_ref_feature()
    else:
        main(cam_id)









    