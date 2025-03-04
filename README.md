# Face Recognition

Python script for face recognition using Yunet for face detection and SFace: Sigmoid-Constrained Hypersphere Loss.

This scripts created for learning purpose. 

For sample code check: https://docs.opencv.org/4.x/df/d20/classcv_1_1FaceDetectorYN.html

For documentation check: https://docs.opencv.org/4.x/df/d20/classcv_1_1FaceDetectorYN.html and https://docs.opencv.org/4.x/da/d09/classcv_1_1FaceRecognizerSF.html


The directory struture for this 
src\
 ├───data\
 ├───images\
 │   ├───Name\
 │   └───Name\
 └───models

## Usage
- Install all required library 
  ``` bash
  pip install -r requirement.txt
  ```
- Create src\data directory (this will store the feature data)
- Create src\models directory (this will store the onnx file)
- Create src\images directory
- Under src\images directory create sub-directory which contain a person name and store that person face image inside that directory
- Download the onnx file, and put under src\models directory
  - [face_detection_yunet_2023mar_int8bq.onnx](https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/face_detection_yunet_2023mar_int8bq.onnx)
  - [face_recognition_sface_2021dec_int8bq.onnx](https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/face_recognition_sface_2021dec_int8bq.onnx)

- Generate feature data
  ```bash
  python main.py -g 
  ```
- Run the application
  ```bash
  python main.py
  ```
  > if you have multiple camera
  ```bash
  python main.py -c 1
  ```

