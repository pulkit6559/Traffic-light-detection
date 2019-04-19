import numpy as np
import os
import six.moves.urllib as urllib
import tarfile
from matplotlib import pyplot as plt
from PIL import Image
from os import path
import time
import cv2

from yolo_model import YOLO
from utils import detect_image

coco_classes = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train',
                'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
                'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed',
                'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard',
                'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
                'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def detect_red(img, Threshold=0.02):
    """
    detect red and yellow
    :param img:
    :param Threshold:
    :return:
    """
  
    desired_dim = (30, 90)  # width, height

    # plt.imshow(img)
    # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    # plt.show()
    img = cv2.resize(img, desired_dim,
                     interpolation=cv2.INTER_LINEAR)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # lower mask (0-10)
    lower_red = np.array([0, 70, 50])
    upper_red = np.array([10, 255, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([170, 70, 50])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    # red pixels' mask
    mask = mask0+mask1

    # Compare the percentage of red values
    rate = np.count_nonzero(mask) / (desired_dim[0] * desired_dim[1])
    print("Rate:   ", rate)
    if rate > Threshold:
        return True
    else:
        return False


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def read_traffic_lights(image, boxes, scores, classes, max_boxes_to_draw=20, min_score_thresh=0.5, traffic_ligth_label=9):
    im_width, im_height = image.size
    red_flag = False
    for i in range(0,min(max_boxes_to_draw, boxes.shape[0])):
        if scores[i] > min_score_thresh and classes[i] == traffic_ligth_label:
            x, y, w, h = tuple(boxes[i].tolist())
            # (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
            #                               ymin * im_height, ymax * im_height)
            crop_img = load_image_into_numpy_array(image.crop((x, y, x+w, y+h)))
            # print("Box  : ", left, " ", right, " ", top, " ", bottom)
            if detect_red(crop_img):
                red_flag = True
                
    print("Red  :", red_flag)
    return red_flag, crop_img


def detect_traffic_lights(PATH_TO_TEST_IMAGES_DIR, Num_images, plot_flag=False):
    """
    Detect traffic lights and draw bounding boxes around the traffic lights
    :param PATH_TO_TEST_IMAGES_DIR: testing image directory
    :param MODEL_NAME: name of the model used in the task
    :return: commands: True: go, False: stop
    """

    # file_data = file.stream.read()
    # nparr = np.fromstring(file_data, np.uint8)
    # img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, '{}'.format(
        i)) for i in os.listdir(PATH_TO_TEST_IMAGES_DIR)]
   

    for image_path in TEST_IMAGE_PATHS:
        Image.MAX_IMAGE_PIXELS = None
        image = Image.open(image_path)
        image_np = load_image_into_numpy_array(image)
        yolo = YOLO(0.6, 0.5)
        # result = detect_image(cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename)), yolo)
        result = detect_image(cv2.imread(image_path), yolo, coco_classes)

        # result will be list of tuple where each tuple is ( class, score, [box coords])
        classes = result[0]
        scores = result[1]
        boxes = result[2]
        # print("Result  :",result)
        # print("Classes  :",classes)
        # print("Boxes  :",boxes)
        # print("Scores  :",scores)

        red_flag, crop_img = read_traffic_lights(image, np.array(boxes), np.array(scores), np.array(classes).astype(np.int32))
        cv2.imwrite('images/res/' + image_path.rsplit('/', 1)
                    [-1], crop_img[..., ::-1])
        if red_flag:
            print('{}: stop'.format(image_path))  # red or yellow
            
        else:
            print('{}: go'.format(image_path))

            
        # print(result[1])
        # img_str = cv2.imencode('.jpg', result[0])[1].tostring()
        # encoded = base64.b64encode(img_str).decode("utf-8")
        # mime = "image/jpg;"
        # out_image = f"data:{mime}base64,{encoded}"

if __name__ == "__main__":
    detect_traffic_lights("images/test",1,True)
