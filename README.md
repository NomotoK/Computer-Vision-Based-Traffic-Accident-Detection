# Computer-Vision-Based-Traffic-Accident-Detection


# Overview
Traffic accidents lead cause of death and injury in our society. The main way of detecting traffic accidents relies on the reporting of people involved or witnesses. Although city monitoring systems have also been used for detection such anomalies, it is difficult to cover every road. In addition, due to the limitations of computing power and algorithms, it usually cannot provide the detecting result in real time. Therefore, this project aims to develop a system that uses computer vision (CV) methods to improve the traffic accidents detection and identification. The system will be capable of analysing videos from different views (monitoring system, mobile phone and driving recorder) so that it can enhance the coverage of monitoring cameras. In addition, this project will optimise the existing CV network structures and utilise edge computing technology to to reduce computational requirements and achieve real-time detection output.

## Objective
- Develop a CV-based system that can determine whether a road accident has occurred by collecting video data from surveillance cameras or dash cam cameras
- The system has an accuracy of over 70% in determining whether a car accident has occurred
- Integrate the system into a desktop or mobile application


![val_batch1_pred](https://github.com/NomotoK/Computer-Vision-Based-Traffic-Accident-Detection/assets/99944622/04ff729d-b386-47de-9b71-b27fa35d387a)




# YOLOv5 requirements

 Usage: pip install -r requirements.txt

## Base 

- gitpython>=3.1.30
- matplotlib>=3.3
- numpy>=1.18.5
- opencv-python>=4.1.1
- Pillow>=7.1.2
- psutil  # system resources
- PyYAML>=5.3.1
- requests>=2.23.0
- scipy>=1.4.1
- thop>=0.1.1  # FLOPs computation
- torch>=1.7.0  # see https://pytorch.org/get-started/locally (recommended)
- torchvision>=0.8.1
- tqdm>=4.64.0
- ultralytics>=8.0.111
- protobuf<=3.20.1   https://github.com/ultralytics/yolov5/issues/8012

## Logging

- tensorboard>=2.4.1
- clearml>=1.2.0
- comet

## Plotting 

pandas>=1.1.4
seaborn>=0.11.0

## Export 

- coremltools>=6.0  # CoreML export
- onnx>=1.10.0  # ONNX export
- onnx-simplifier>=0.4.1  # ONNX simplifier
- nvidia-pyindex  # TensorRT export
- nvidia-tensorrt  # TensorRT export
- scikit-learn<=1.1.2  # CoreML quantization
- tensorflow>=2.4.0  # TF exports (-cpu, -aarch64, -macos)
- tensorflowjs>=3.9.0  # TF.js export
- openvino-dev  # OpenVINO export

## Deploy

setuptools>=65.5.1 # Snyk vulnerability fix
= tritonclient[all]~=2.24.0

## Extras

- ipython   interactive notebook
- mss  # screenshots
- albumentations>=1.0.3
- pycocotools>=2.0.6  # COCO mAP
