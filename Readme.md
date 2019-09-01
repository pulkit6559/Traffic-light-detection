### Steps to run the scripts

1. Download and convert the weights for yolov3

`wget https://pjreddie.com/media/files/yolov3.weights`

`python3 yad2k.py yolov3.cfg yolov3.weights yolo.h5`

2. Paste the images on which traffic light detection is to be performed in `images/test` directory

3. To view the results of YOLOv3 algorithms for object detection run script utils.py

`python3 utils.py`

4. To test traffic light detection run script red_light_detection.py

`python3 red_light_detection.py`

The results will be displayed in the terminal against the name of every test image, to view the cropped images of traffic light browse thwough the `images/res` folder 
