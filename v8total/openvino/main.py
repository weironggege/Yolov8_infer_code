'''
pip install openvino

yolo export model=yolov8n.pt format=openvino
'''
import cv2
import numpy as np

from openvino.runtime import Core


class OpenVYolov8:

    def __init__(self, conf_thres=0.7, nms_thres=0.8):
        
        self.confThreshold = conf_thres
        self.nmsThreshold = nms_thres

        self.model_xml = "yolov8n_openvino_model/yolov8n.xml"
        self.model_bin = "yolov8n_openvino_model/yolov8n.bin"

        self.class_names = [x.strip() for x in open("coco.names", "r").readlines()]

        self.ie = Core()
        self.net = self.ie.read_model(model=self.model_xml, weights=self.model_bin)
        self.device = "CPU"
        self.complied_model = self.ie.compile_model(model=self.net, device_name=self.device)
        self.output_layer = self.complied_model.output(0)

        self.inpWidth = 640
        self.inpHeight = 640


    def drawPred(self, image, box, conf, clsid):

        x,y,w,h = box

        cv2.rectangle(image, (x,y), (x+w,y+h), (178,45,233), 2)

        label = self.class_names[clsid] + ":" + str(round(conf * 100)) + "%"

        cv2.putText(image, label, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (233,0,45), 2)

    def resize_img(self, image):

        neww, newh, padw, padh = self.inpWidth, self.inpHeight, 0, 0
        srch, srcw = image.shape[:2]
        if srch != srcw:
            hw_scale = srch / srcw
            if hw_scale > 1.0:
                neww = int(self.inpWidth / hw_scale)
                simg = cv2.resize(image, (neww, newh), cv2.INTER_AREA)
                padw = int((self.inpWidth - neww) * 0.5)
                simg = cv2.copyMakeBorder(simg, 0, 0, padw, self.inpWidth-neww-padw, cv2.BORDER_CONSTANT, (114,114,114))
            else:
                newh = int(self.inpHeight * hw_scale)
                simg = cv2.resize(image, (neww, newh), cv2.INTER_AREA)
                padh = int((self.inpHeight - newh) * 0.5)
                simg = cv2.copyMakeBorder(simg, padh, self.inpHeight-newh-padh, 0, 0, cv2.BORDER_CONSTANT, (114,114,114))
        else:
            simg = cv2.resize(image, (neww, newh), cv2.INTER_AREA)

        return simg, neww, newh, padw, padh


    def detect(self, image):

        simg, neww, newh, padw, padh = self.resize_img(image)
        blob = cv2.dnn.blobFromImage(simg, 1.0 / 255.0, swapRB=True)

        results = self.complied_model([blob])[self.output_layer]

        ratiow, ratioh = image.shape[1] / neww, image.shape[0] / newh
        boxes, confs, ids = [], [], []

        for pred in np.squeeze(results).transpose():

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
    
    srcimg = cv2.imread("imgs/bus.jpg")

    yolonet = OpenVYolov8()

    tarimg = yolonet.detect(srcimg)

    cv2.imwrite("imgs/bus_openvino.jpg", tarimg)


        
