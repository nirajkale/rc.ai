from typing import Any
import tensorrt as trt
import numpy as np
import cv2
from vision_utils import (draw_and_collect_bbox, \
    non_max_suppression, scale_coords, \
    preprocess, Detection)
import torch
import yaml
from PIL import Image
from collections import OrderedDict, namedtuple
from hardware_utils import select_device
from typing import List, Tuple
import warnings
from os import path

Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
logger = trt.Logger(trt.Logger.INFO)

class ObjectDetector:

    def __init__(self, engine_path, input_size=640, labels:List[str]=None, **kwargs) -> None:
        self.device = select_device(kwargs.get('device', 0))
        print('selected device for inference:', self.device)
        print('de-serialing tensorrt engine')
        with open(engine_path, 'rb') as f, trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(f.read())
        print('loading engine bindings & allocating device memory')
        self.bindings = OrderedDict()
        for index in range(model.num_bindings):
            name = model.get_binding_name(index)
            dtype = trt.nptype(model.get_binding_dtype(index))
            shape = tuple(model.get_binding_shape(index))
            data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(self.device)
            self.bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
        self.context = model.create_execution_context()
        print('engine formalities completed')
        self.imgsz = (input_size, input_size)
        self.labels = labels
        self.nc = len(self.labels) if self.labels else 0
        #load optional settings
        self.conf_thres = kwargs.get('conf_thres', 0.25)
        self.half = kwargs.get('half', None)
        self.iou_thres = kwargs.get('iou_thres', 0.45)
        self.classes = kwargs.get('classes', None)
        self.agnostic_nms = kwargs.get('agnostic_nms', False)
        self.max_detections = kwargs.get('max_detections', 100)
        self.warn = kwargs.get('warn', True)
        self.warmup = kwargs.get('warmup', True)
        if self.half is None and 'fp16' in path.basename(engine_path).lower():
            self.half = True
        if self.warmup and isinstance(self.device, torch.device) and self.device.type != 'cpu':  # only warmup GPU models
            print('warming up engine on ', self.device)
            im = torch.zeros((1, 3, 640, 640)).to(self.device).type(torch.half if self.half else torch.float)  # input image
            self.forward(im)  # warmup
        print('precision is set to: ', ('fp16' if self.half else 'fp32'))
        print('ready to infer')

    def load_coco_labels(self, yaml_path:str=r"models/coco128.yaml"):
        with open(yaml_path, errors='ignore') as f:
            self.labels = yaml.safe_load(f)['names']  # class names
        self.nc = len(self.labels) if self.labels else 0

    def forward(self, im):
        assert im.shape == self.bindings['images'].shape, (im.shape, self.bindings['images'].shape)
        self.binding_addrs['images'] = int(im.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        return self.bindings['output'].data

    def predict(self, img0, isBGR=False, scaled_inference=False)-> Tuple[np.ndarray, List[Detection]]:
        if isBGR:
            img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        if scaled_inference:
            original_image = np.copy(img0)
        if img0.shape[:2]!= self.imgsz:
            if self.warn:
                # warnings.warn('Image is being shaped, check gstreamer pipeline settings')
                raise Exception('Image is being shaped, check gstreamer pipeline settings')
            img0 = preprocess(img0, new_shape = self.imgsz, stride=32)[0]
        img = np.copy(img0)
        img = img.transpose((2, 0, 1))[None]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255
        y = self.forward(img)
        det = non_max_suppression(y, self.nc)
        if isinstance(det, torch.Tensor):
            det = det.cpu().numpy()
        if det.shape[0]==0:
            return original_image if scaled_inference else img0, []
        elif scaled_inference:
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], original_image.shape).round()
            return draw_and_collect_bbox(original_image, det, self.labels)
        else:
            return draw_and_collect_bbox(img0, det, self.labels)

    @staticmethod
    def save_np_image(image, output_path):
        image = Image.fromarray(image)
        image.save(output_path)

if __name__ == '__main__':


    engine_path = r"/home/niraj/projects/yolov5/yolov5n_fp16.engine"
    detector = ObjectDetector(engine_path, half=False)
    detector.load_coco_labels()
    img0 = cv2.imread(r"person.jpg")
    img1, detections = detector.predict(img0, isBGR=True, scaled_inference= True)
    detector.save_np_image(img1, "output.jpg")
    print('done')