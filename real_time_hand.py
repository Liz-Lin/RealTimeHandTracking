import numpy as np
import sys
import tensorflow as tf
import os
from threading import Thread
from datetime import datetime
import cv2
from utils import label_map_util
from collections import defaultdict
from utils import visualization_utils as vis_util

detection = tf.Graph()
_score_thresh = 0.30

MODEL_NAME = 'ssd'

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join("training", 'hand_detection.pbtxt')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=1, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_inference():
    detection = tf.Graph()
    with detection.as_default():
        graph = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as file:
            serialized_graph = file.read()
            graph.ParseFromString(serialized_graph)
            tf.import_graph_def(graph, name='')
        sess = tf.Session(graph=detection)
    return detection, sess

def detect_obj(image_np,detection,sess):
    image_tenor = detection.get_tensor_by_name('image_tensor:0')
    box = detection.get_tensor_by_name('detection_boxes:0')
    score = detection.get_tensor_by_name('detection_scores:0')
    classe = detection.get_tensor_by_name('detection_classes:0')
    num = detection.get_tensor_by_name('num_detections:0')
    np_expand = np.expand_dims(image_np,axis=0)
    (boxes,scores,classes,nums) = sess.run([box, score, classe, num],
                                           feed_dict={image_tenor:np_expand})
    vis_util.visualize_boxes_and_labels_on_image_array(
                                                   image_np,
                                                   np.squeeze(boxes),
                                                   np.squeeze(classes).astype(np.int32),
                                                   np.squeeze(scores),
                                                   category_index,
                                                   use_normalized_coordinates=True,
                                                   line_thickness=8)
    return

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
    detection, sess = load_inference()
    while True:
        ret, image_np = cap.read()
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        detect_obj(image_np, detection,sess)
        cv2.imshow('Hand Detection', cv2.cvtColor(image_np,cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destoryAllWindows()
            bereak









