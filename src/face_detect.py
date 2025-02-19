# face_detect.py
# To detect face in a webcam

import cv2
import numpy as np  

class FaceDetect:
    def __init__(self, model_path, config = "", input_size=tuple([320,320]), score_threshold = 0.9, nms_threshold=0.3, topk_threshold=5000, backend_id=0, target_id=0):
        self._model_path = model_path
        self._config = config
        self._input_size = input_size
        self._score_threshold = score_threshold
        self._nms_threshold = nms_threshold
        self._topk_threshold = topk_threshold
        self._backend_id = backend_id
        self._target_id = target_id

        self._detector = cv2.FaceDetectorYN.create(self._model_path,
                                                    self._config, 
                                                    self._input_size, 
                                                    self._score_threshold, 
                                                    self._nms_threshold, 
                                                    self._topk_threshold, 
                                                    self._backend_id, 
                                                    self._target_id)

    @property
    def name(self) :
        return self.__class__.__name__
        
    def detect_face(self, frame) :
        results = self._detector.detect(frame)
        return results
    
    def set_input_size(self, input_size):
        self._detector.setInputSize(tuple(input_size))

        
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # Get the frame width
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Get the frame height

    is_flipped = False # Flag to check whether the frame already flipped or not

    tm = cv2.TickMeter()
    detector = FaceDetect(model_path='models/face_detection_yunet_2023mar_int8bq.onnx', input_size=[w, h])

    while True:
        ret, frame = cap.read()
        if not ret :
            print("No frame captured")
            break
        frame = cv2.flip(frame, 1)
        tm.start()
        results = detector.detect_face(frame)
        tm.stop()
        
        
        if results[1] is not None:
            for idx, result in enumerate(results[1]):
                print('Face {}, top-left coordinates: ({:.2f}, {:.2f}), box width: {:.2f}, box height {:.2f}, score: :{:.2f}'.format(idx,result[0], result[1], result[2], result[3], result[-1]))            
                
                # coord = result[:-1].astype(np.int32)  # convert the coordinate value to integer
                coord = tuple(map(np.int32, result[:-1]))
                # new_x = w-coord[0]-coord[2] # Need to convert the new x coordinate after we flip the frame
                
                cv2.rectangle(frame, (coord[0], coord[1]), (coord[0]+coord[2],coord[1]+coord[3]), (0,255,0), 2) # Draw the rectangle arround the face
                
                cv2.circle(frame, (coord[4], coord[5]), 2, (0,0,255), 2) # Right eye marker
                cv2.circle(frame, (coord[6], coord[7]), 2, (0,255,0), 2) # Left eye marker
                cv2.circle(frame, (coord[8], coord[9]), 2, (0,110,255), 2) # Nose tip marker
                cv2.circle(frame, (coord[10], coord[11]), 2, (0,255,255), 2) # Mouth right corner marker
                cv2.circle(frame, (coord[12], coord[13]), 2, (255,255,0), 2) # Mouth left corner marker

                # # Draw line to connect each marker, where tip of nose as centre
                cv2.line(frame, (coord[4], coord[5]),(coord[8], coord[9]), (255,0,182),2 ) # Right eye - Nose tip
                cv2.line(frame, (coord[6], coord[7]),(coord[8], coord[9]), (255,0,182), 2 ) # Left eye - Nose tip
                cv2.line(frame, (coord[10], coord[11]),(coord[8], coord[9]), (255,0,182), 2) # Mouth Right corner - Nose tip
                cv2.line(frame, (coord[12], coord[13]),(coord[8], coord[9]), (255,0,182), 2 ) # Mouth left corner - Node tip

                cv2.putText(frame, "Confidence Score: {:.2f}".format(result[-1]), (coord[0], coord[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2 )  
        else:            
            cv2.putText(frame, "Face not found", (1, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2 )
            print("Face not found")
        
        cv2.putText(frame, "FPS: {:.2f}".format(tm.getFPS()), (1,16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2 )
        cv2.imshow("Face Recognition", frame)

        pressed_key = cv2.waitKey(1) & 0xFF
        if pressed_key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    

    

    

    