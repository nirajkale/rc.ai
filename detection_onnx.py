import warnings
import cv2
import numpy as np
import onnxruntime as rt
from PIL import Image
import torch
from torch._C import dtype
from vision_utils import (draw_and_collect_bbox, \
    non_max_suppression, scale_coords, \
    preprocess, Detection)
import time
import yaml
from typing import Tuple, List

class ObjectDetector:

    onnx_providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']

    def __init__(self, onnx_model_path:str, \
            input_size:int = 640,\
            labels:List[str]=None, \
            providers= onnx_providers, **kwargs) -> None:
        start = time.time()
        print('loading onnx model')
        # ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        self.sess = rt.InferenceSession(onnx_model_path, providers=providers)
        self.output_name = self.sess.get_outputs()[0].name
        self.input_name = self.sess.get_inputs()[0].name
        print(f"model loaded in {int(time.time() - start)} seconds")
        self.imgsz = (input_size, input_size)
        self.labels = labels
        self.nc = len(self.labels) if self.labels else 0
        #load optional settings
        self.conf_thres = kwargs.get('conf_thres', 0.25)
        self.iou_thres = kwargs.get('iou_thres', 0.45)
        self.classes = kwargs.get('classes', None)
        self.agnostic_nms = kwargs.get('agnostic_nms', False)
        self.max_detections = kwargs.get('max_detections', 100)

    def load_coco_labels(self, yaml_path:str=r"models/coco128.yaml"):
        with open(yaml_path, errors='ignore') as f:
            self.labels = yaml.safe_load(f)['names']  # class names
        self.nc = len(self.labels) if self.labels else 0

    def perform_warmup(self, steps = 3):
        print('warming up inference session')
        img0 = np.zeros((1, 3,)+ self.imgsz, dtype='float32')
        for _ in range(steps):
            _ = self.sess.run([self.output_name], {self.input_name: img0})[0]
        print('warm up done')

    def predict(self, img0, isBGR=False, scaled_inference=False)-> Tuple[np.ndarray, List[Detection]]:
        if isBGR:
            img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        if scaled_inference:
            original_image = np.copy(img0)
        if img0.shape[:2]!= self.imgsz:
            warnings.warn('Image is being shaped, check gstreamer pipeline settings')
            img0 = preprocess(img0, new_shape = self.imgsz, stride=32)[0]
        img = np.copy(img0)#check: if np.ascontigious array could help here
        img = img.transpose((2, 0, 1))/255  # HWC to CHW
        img = np.expand_dims(img, axis=0).astype('float32')
        prediction = self.sess.run([self.output_name], {self.input_name: img})[0]
        #postprocess (1, 25200, 85)
        pred = torch.tensor(prediction[0])
        det = non_max_suppression(pred, self.nc, \
            self.conf_thres, self.iou_thres, \
            self.classes, self.agnostic_nms, max_det=self.max_detections)
        if det.shape[0]==0:
            return original_image if scaled_inference else img, []
        elif scaled_inference:
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], original_image.shape).round()
            return draw_and_collect_bbox(original_image, det, self.labels)
        else:
            return draw_and_collect_bbox(img, det, self.labels)

    @staticmethod
    def save_np_image(image, output_path):
        image = Image.fromarray(image)
        image.save(output_path)


if __name__ == '__main__':

    img0 = cv2.imread(r"person.jpg")
    model_path = r"/home/niraj/projects/yolov5/yolov5s.onnx"
    detector = ObjectDetector(model_path, providers=["CPUExecutionProvider"])
    detector.load_coco_labels()
    img1, detections = detector.predict(img0, isBGR=True, scaled_inference= True)
    detector.save_np_image(img1, "output.jpg")
    print('done')