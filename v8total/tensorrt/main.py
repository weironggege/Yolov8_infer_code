'''
install tensorrt
yolo export model=yolov8n.pt format=engine
'''

import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import json

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


class YOLOV8RT:

    def __init__(self, modelpath, conf_thres=0.7, nms_thres=0.8):

        self.confThreshold = conf_thres
        self.nmsThreshold = nms_thres

        self.class_names = [x.strip() for x in open("coco.names", "r").readlines()]


        self.inpWidth = 640
        self.inpHeight = 640

        with open(modelpath, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            try:
                meta_len = int.from_bytes(f.read(4), byteorder="little")  # read metadata length
                metadata = json.loads(f.read(meta_len).decode("utf-8"))  # read metadata
            except UnicodeDecodeError:
                f.seek(0) 
            self.engine = runtime.deserialize_cuda_engine(f.read())
        

    def drawPred(self, image, box, conf, clsid):

        x,y,w,h = box

        cv2.rectangle(image, (x,y), (x+w,y+h), (89,67,0), 2)

        label = self.class_names[clsid] + ":" + str(round(conf * 100)) + "%"

        cv2.putText(image, label, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (233,0,156), 2)


    def resize_img(self, image):
        neww, newh, padw, padh = self.inpWidth, self.inpHeight, 0, 0

        srch, srcw = image.shape[:2]

        if srch != srcw:
            hw_scale = srch / srcw
            if hw_scale > 1.0:
                neww = int(self.inpWidth / hw_scale)
                simg = cv2.resize(image, (neww,newh), cv2.INTER_AREA)
                padw = int((self.inpWidth - neww) * 0.5)
                simg = cv2.copyMakeBorder(simg, 0, 0, padw, self.inpWidth-neww-padw, cv2.BORDER_CONSTANT, (114,114,114))
            else:
                newh = int(self.inpHeight * hw_scale)
                simg = cv2.resize(image, (neww,newh), cv2.INTER_AREA)
                padh = int((self.inpHeight - newh) * 0.5)
                simg = cv2.copyMakeBorder(simg, padh, self.inpHeight-newh-padh, 0, 0, cv2.BORDER_CONSTANT, (114,114,114))
        else:
            simg = cv2.resize(image, (neww,newh), cv2.INTER_AREA)
        return simg, neww, newh, padw, padh

    def detect(self, image):

        simg, neww, newh, padw, padh = self.resize_img(image)

        blob = cv2.dnn.blobFromImage(simg, 1.0 / 255.0, swapRB=True)

        context = self.engine.create_execution_context()

        input_binding_name = self.engine.get_binding_index("images")
        output_binding_name = self.engine.get_binding_index("output")
        
        N, C, H, W = 1, 3, self.inpHeight, self.inpWidth

        d_input = cuda.mem_alloc(N * C * H * W * np.dtype(np.float32).itemsize)
        d_output = cuda.mem_alloc(N * 84 * 8400 * np.dtype(np.float32).itemsize)

        bindings = [int(d_input), int(d_output)]

        stream = cuda.Stream()
        cuda.memcpy_htod_async(d_input, blob, stream)
        stream.synchronize()

        context.execute(N, bindings)
        stream.synchronize()

        out = np.empty((N, 84, 8400), dtype=np.float32)
        cuda.memcpy_dtoh_async(out, d_output, stream)
        stream.synchronize()

        ratiow, ratioh = image.shape[1] / neww, image.shape[0] / newh
        boxes, confs, ids = [], [], []

        for pred in np.squeeze(out).transpose():

            maxss = np.max(pred[4:])
            idp = np.argmax(pred[4:])

            if maxss >= self.confThreshold:
                cx = (pred[0] - padw) * ratiow
                cy = (pred[1] - padh) * ratioh
                w = pred[2] * ratiow
                h = pred[3] * ratioh

                left = int(cx - 0.5 * w)
                top = int(cy - 0.5 * h)

                boxes.append([left, top, int(w), int(h)])
                confs.append(maxss)
                ids.append(idp)

        indices = cv2.dnn.NMSBoxes(boxes, confs, self.confThreshold, self.nmsThreshold)

        for idx in indices:
            self.drawPred(image, boxes[idx], confs[idx], ids[idx])
        return image

if __name__ == "__main__":



    rtmodel = YOLOV8RT("weights/yolov8n.engine")
    
    
    srcimg = cv2.imread("imgs/bus.jpg")

    tarimg = rtmodel.detect(srcimg)

    cv2.imwrite("imgs/bus_rt.jpg", tarimg)



