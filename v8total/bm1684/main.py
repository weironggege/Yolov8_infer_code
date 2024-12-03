'''
export yolov8n onnx
install sophon api
trans onnx .bmodel
'''
import cv2
import numpy as np
import sophon.sail as sail

class YOLOV8:

    def __init__(self, modelpath, conf_thres=0.7, nms_thres=0.8):

        self.confThreshold = conf_thres
        self.nmsThreshold = nms_thres

        self.class_names = [x.strip() for x in open("coco.names", "r").readlines()]

        self.inpWidth = 640
        self.inpHeight = 640
        
        self.io_mode = sail.IOMode.SYSIO
        self.device_id = 0
        
        self.engine = sail.Engine(modelpath, self.device_id, self.io_mode)
        
        self.input_names = self.engine.get_input_names(self.engine.get_graph_names()[0])
        self.output_names = self.engine.get_output_names(self.engine.get_graph_names()[0])

    def drawPred(self, image, box, classid, conf):
        x,y,w,h = box

        cv2.rectangle(image, (x, y), (x+w, y+h), (255,0,0), 2)

        label = self.class_names[classid] + ":" + str(round(conf * 100)) + "%"

        cv2.putText(image, label, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    def resize_img(self, image):
        neww, newh, padw, padh = self.inpWidth, self.inpHeight, 0, 0

        srch, srcw = image.shape[:2]

        if srch != srcw:
            hw_scale = srch / srcw
            if hw_scale > 1.0:
                neww = int(self.inpWidth / hw_scale)
                timg = cv2.resize(image, (neww, newh), CV2.INTER_AREA)
                padw = int((self.inpWidth - neww) * 0.5)
                timg = cv2.copyMakeBorder(timg, 0, 0, padw, self.inpWidth-neww-padw, cv2.BORDER_CONSTANT, (114,114,114))
            else:
                newh = int(self.inpHeight * hw_scale)
                timg = cv2.resize(image, (neww, newh), cv2.INTER_AREA)
                padh = int((self.inpHeight - newh) * 0.5)
                timg = cv2.copyMakeBorder(timg, padh, self.inpHeight-newh-padh, 0, 0, cv2.BORDER_CONSTANT, (114,114,114))
        else:
            timg = cv2.resize(image, (neww, newh), cv2.INTER_AREA)

        return timg, neww, newh, padw, padh

    def detect(self, image):

        timg, neww, newh, padw, padh = self.resize_img(image)

        blob = cv2.dnn.blobFromImage(timg, 1.0 / 255.0, swapRB=True)

        input_feed = {self.input_names[0]:np.ascontiguousarray(blob)}

        outs = self.engine.process(self.engine.get_graph_names()[0], input_feed)

        ratioh, ratiow = image.shape[0] / newh, image.shape[1] / neww
        boxes, ids, confs = [], [], []

        for pred in np.squeeze(outs[self.output_names[0]]).transpose():
            
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
            self.drawPred(image, boxes[idx], ids[idx], confs[idx])
        return image

if __name__ == "__main__":

    net = YOLOV8("weights/yolov8n_fp32.bmodel")

    srcimg = cv2.imread("imgs/person.jpg")

    tarimg = net.detect(srcimg)

    cv2.imwrite("imgs/person_bm_py.jpg", tarimg)
