from typing import Any
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
from vision_utils import (draw_and_collect_bbox, \
    non_max_suppression, scale_coords, \
    preprocess, Detection)
from dataclasses import dataclass
import torch
import yaml
from PIL import Image

@dataclass
class HostDeviceMem:
    host_mem:Any
    device_mem:Any
    name:str=None
    dtype:Any=None
    shape:Any=None

def allocate_buffers(model):
    input_buffer = None
    output_buffer = None
    bindings = []
    stream = cuda.Stream()
    for index, binding in enumerate(model):
        size = trt.volume(model.get_binding_shape(binding)) * model.max_batch_size
        dtype = trt.nptype(model.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        name = model.get_binding_name(index)
        dtype = trt.nptype(model.get_binding_dtype(index))
        shape = tuple(model.get_binding_shape(index))
        # Append to the appropriate list.
        if model.binding_is_input(binding):
            input_buffer = HostDeviceMem(host_mem, device_mem, name, dtype, shape)
        elif name=='output':
            output_buffer = HostDeviceMem(host_mem, device_mem, name, dtype, shape)
    return input_buffer, output_buffer, bindings, stream

def do_inference(context, bindings, input_buffer:HostDeviceMem, output_buffer:HostDeviceMem, stream, batch_size=1):
    # Transfer input data to the GPU.
    # [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    cuda.memcpy_htod_async(input_buffer.device_mem, input_buffer.host_mem, stream)
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    # [cuda.memcpy_dtoh_async(output_buffer.host_mem, output_buffer.device_mem, stream) for out in outputs]
    cuda.memcpy_dtoh_async(output_buffer.host_mem, output_buffer.device_mem, stream)
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    # return [out.host for out in outputs]
    return output_buffer.host_mem

def load_coco_labels(yaml_path:str=r"models/coco128.yaml"):
    with open(yaml_path, errors='ignore') as f:
        return yaml.safe_load(f)['names']

logger = trt.Logger(trt.Logger.INFO)
model_path = r'models/yolov5n6_fp16.engine'
with open(model_path, 'rb') as f, trt.Runtime(logger) as runtime:
    model = runtime.deserialize_cuda_engine(f.read())

input_buffer, output_buffer, bindings, stream = allocate_buffers(model)
context = model.create_execution_context()

img0 = cv2.imread(r"yolov4/kite.jpg")
img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
img0 = preprocess(img0, new_shape = (640, 640), stride=32)[0]
img = np.copy(img0)#check: if np.ascontigious array could help here
img = img.transpose((2, 0, 1))/255  # HWC to CHW
# img = np.expand_dims(img, axis=0).astype('float32')
# img_np = img.ravel()
np.copyto(input_buffer.host_mem, img.ravel())
prediction = do_inference(context, bindings, input_buffer, output_buffer, stream=stream)
pred = torch.tensor(prediction[0])

classes = load_coco_labels()

det = non_max_suppression(pred, 80, \
    0.25, 0.45, \
    classes, False, max_det=100)

img = draw_and_collect_bbox(img, det, classes)
image = Image.fromarray(img)
image.save(img, "output.jpg")

print('done')