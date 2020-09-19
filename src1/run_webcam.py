import argparse
import logging
import time

import cv2
import numpy as np

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh
import json
logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

def start():
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--zoom', type=float, default=1.0)
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--output_json', type=str, default='3d-pose-baseline/src3d/json/', help='writing output json dir')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    logger.debug('cam read+')
    cam = cv2.VideoCapture("http://127.0.0.1:8000/")
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))
    frame=0
    while True:
        ret_val, image = cam.read()
        #image2 = cv2.threshold(image,0,255,cv2.THRESH_BINARY)
        image2 = cv2.imread('../images/123.jpg')
        logger.debug('image preprocess+')
        if args.zoom < 1.0:
            canvas = np.zeros_like(image)
            canvas2 = np.zeros_like(image2)
            img_scaled = cv2.resize(image, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_LINEAR)
            img_scaled2 = cv2.resize(image2, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_LINEAR)
            dx = (canvas.shape[1] - img_scaled.shape[1]) // 2
            dy = (canvas.shape[0] - img_scaled.shape[0]) // 2
            canvas[dy:dy + img_scaled.shape[0], dx:dx + img_scaled.shape[1]] = img_scaled
            canvas2[dy:dy + img_scaled2.shape[0], dx:dx + img_scaled2.shape[1]] = img_scaled2
            image = canvas
            image2 = canvas2
        elif args.zoom > 1.0:
            img_scaled = cv2.resize(image, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_LINEAR)
            dx = (img_scaled.shape[1] - image.shape[1]) // 2
            dy = (img_scaled.shape[0] - image.shape[0]) // 2
            image = img_scaled[dy:image.shape[0], dx:image.shape[1]]
        #image2=np.array(image2)
        logger.debug('image process+')
        humans = e.inference(image)
        n = len(humans)
        print(humans)
        """body = {}
        body["key_points"] = []
        for human in humans:
           for parts in human.body_parts.items():
               print("id" ,parts[0])
               print("x",parts[1].x)
               print("y",parts[1].y)
               print("score = ",parts[1].score)
               body["key_points"].append({"ID":parts[0],"X":parts[1].x,"Y":parts[1].y})
        with open("json/{0}.json".format(str(i)),'w') as file:
        	json.dump(body,file)"""
        logger.debug('postprocess+')
        image2 = TfPoseEstimator.draw_humans(image2, humans, imgcopy=False,frame=frame,dir=args.output_json)
       # image =  
        logger.debug('show+')
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.putText(image, "persons = %f" % n,(10,30),cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 3)
        # cv2.imshow('orignal',image)
        # cv2.imshow('tf-pose-estimation result', image2)
        frame=frame+1
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break
        logger.debug('finished+')
        return image ,image2

    cv2.destroyAllWindows()


if __name__ == '__main__':
    start()
