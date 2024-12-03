'''
pip install ncnn
yolo export model=yolov8s.pt format=ncnn
'''
import time
import numpy as np
import ncnn
# from .model_store import get_model_file
# from ..utils.objects import Detect_Object
# from ..utils.functional import *
from typing import Iterable
import ncnn
from ncnn.model_zoo.model_store import get_model_file
from ncnn.utils.objects import Detect_Object
from ncnn.utils.functional import *
import cv2


class YoloV8s:
    def __init__(
        self,
        target_size=640,
        prob_threshold=0.1,
        nms_threshold=0.45,
        num_threads=1,
        use_gpu=False,
    ):
        self.target_size = target_size
        self.prob_threshold = prob_threshold
        self.nms_threshold = nms_threshold
        self.num_threads = num_threads
        self.use_gpu = use_gpu
        self.inpWidth = 640
        self.inpHeight = 640

        self.reg_max = 16
        self.mean_vals = []
        self.norm_vals = [1 / 255.0, 1 / 255.0, 1 / 255.0]

        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = self.use_gpu
        self.net.opt.num_threads = self.num_threads

        # original pretrained model from https://github.com/ultralytics/ultralytics
        # the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
        self.net.load_param("./weights/v8safehat.param")
        self.net.load_model("./weights/v8safehat.bin")

        self.grid = [make_grid(20, 20), make_grid(40, 40), make_grid(80, 80)]
        self.stride = np.array([32, 16, 8])
        
        '''
        self.class_names = [
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "dining table",
            "toilet",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
        ]
        '''
        self.class_names = ["hat", "head", "other_hat"]
    
    def drawPred(self, image, box, clsid, conf):
        x,y,w,h = box

        cv2.rectangle(image, (x,y), (x+w,y+h), (0,114,123), 2)

        label = self.class_names[clsid] + ":" + str(round(conf * 100)) + "%"

        cv2.putText(image, label, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    def resize_img(self, image):
        neww, newh, padw, padh = self.inpWidth, self.inpHeight, 0, 0

        srch, srcw = image.shape[:2]

        if srch != srcw:
            hw_scale = srch / srcw
            if hw_scale > 1.0:
                neww = int(self.inpWidth / hw_scale)
                timg = cv2.resize(image, (neww, newh), cv2.INTER_AREA)
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


    def __del__(self):
        self.net = None

    def __call__(self, img):
        img_w = img.shape[1]
        img_h = img.shape[0]
        image = img
        timg, neww, newh, wpad, hpad = self.resize_img(image)
        '''
        w = img_w
        h = img_h
        scale = 1.0
        if w > h:
            scale = float(self.target_size) / w
            w = self.target_size
            h = int(h * scale)
        else:
            scale = float(self.target_size) / h
            h = self.target_size
            w = int(w * scale)
        '''
        mat_in = ncnn.Mat.from_pixels_resize(
            img, ncnn.Mat.PixelType.PIXEL_BGR2RGB, img_w, img_h, neww, newh
        )
        # pad to target_size rectangle
        # yolov5/utils/datasets.py letterbox
        # wpad = (w + 31) // 32 * 32 - w
        # hpad = (h + 31) // 32 * 32 - h
        mat_in_pad = ncnn.copy_make_border(
            mat_in,
            hpad,
            self.inpHeight - newh - hpad,
            wpad,
            self.inpWidth - neww - wpad,
            ncnn.BorderType.BORDER_CONSTANT,
            114.0,
        )

        mat_in_pad.substract_mean_normalize(self.mean_vals, self.norm_vals)

        ex = self.net.create_extractor()
        ex.input("in0", mat_in_pad)

        ret1, mat_out1 = ex.extract("out0")  # stride 8
        ratioh, ratiow = image.shape[0] / newh, image.shape[1] / neww
        boxes, confs, ids = [], [], []
        
        print(np.array(np.unsqueeze(mat_out1)).shape)
        for pred in np.array(mat_out1).transpose():
            maxss = np.max(pred[4:])
            idp = np.argmax(pred[4:])
            if maxss >= self.prob_threshold:
                cx = (pred[0] - wpad) * ratiow
                cy = (pred[1] - hpad) * ratioh
                w = pred[2] * ratiow
                h = pred[3] * ratioh

                left = int(cx - 0.5 * w)
                top = int(cy - 0.5 * h)
                

                boxes.append([left, top, int(w), int(h)])
                confs.append(maxss)
                ids.append(idp)

        indices = cv2.dnn.NMSBoxes(boxes, confs, self.prob_threshold, self.nms_threshold)

        for idx in indices:
            self.drawPred(image, boxes[idx], ids[idx], confs[idx])
        
        return image

        '''
        ret2, mat_out2 = ex.extract("out1")  # stride 16
        ret3, mat_out3 = ex.extract("out2")  # stride 32

        # pred = [np.array(mat_out1)]
        # print(pred[0].shape)
        pred = [np.array(mat_out3), np.array(mat_out2), np.array(mat_out1)]
        print(pred[0].shape, pred[1].shape, pred[2].shape)
        z = []
        for i in range(len(pred)):
            num_grid_x = mat_in_pad.w // self.stride[i]
            num_grid_y = mat_in_pad.h // self.stride[i]
            if (
                    self.grid[i].shape[1] != num_grid_y
                    or self.grid[i].shape[2] != num_grid_x
            ):
                self.grid[i] = make_grid(num_grid_x, num_grid_y)
            cls, box = np.split(pred[i].transpose((1, 2, 0)), [len(self.class_names), ], -1)
            box = softmax(box.reshape(-1, self.reg_max))
            box = box.reshape(num_grid_y, num_grid_x, 4, self.reg_max)
            box = box @ np.arange(0, self.reg_max, dtype=np.float32)
            cls = sigmoid(cls)
            conf = cls.max(-1, keepdims=True)
            x1y1 = (self.grid[i][0] + 0.5 - box[..., :2]) * self.stride[i]
            x2y2 = (self.grid[i][0] + 0.5 + box[..., 2:]) * self.stride[i]
            res = np.concatenate([x1y1, x2y2, conf, cls], -1)
            z.append(res.reshape((1, -1, len(self.class_names) + 5)))
        pred = np.concatenate(z, 1)

        result = self.non_max_suppression(
            pred, self.prob_threshold, self.nms_threshold
        )[0]

        if isinstance(result, Iterable):
            objects = [
                Detect_Object(
                    obj[5],
                    obj[4],
                    (obj[0] - (wpad / 2)) / scale,
                    (obj[1] - (hpad / 2)) / scale,
                    (obj[2] - obj[0]) / scale,
                    (obj[3] - obj[1]) / scale,
                )
                for obj in result
            ]
        else:
            objects = []

        return objects
        '''

    def non_max_suppression(
        self,
        prediction,
        conf_thres=0.1,
        iou_thres=0.6,
        merge=False,
        classes=None,
        agnostic=False,
    ):
        """Performs Non-Maximum Suppression (NMS) on inference results

        Returns:
            detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
        """
        nc = prediction[0].shape[1] - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        # Settings
        min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
        max_det = 300  # maximum number of detections per image
        time_limit = 10.0  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

        t = time.time()
        output = [None] * prediction.shape[0]
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            # If none remain process next image
            if not x.shape[0]:
                continue

            box = x[:, :4]

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:] > conf_thres).nonzero()
                x = np.concatenate(
                    (box[i], x[i, j + 5, None], j[:, None].astype(np.float32)), axis=1
                )
            else:  # best class only
                conf, j = x[:, 5:].max(1, keepdim=True)
                x = np.concatenate((box, conf, j.float()), axis=1)[
                    conf.view(-1) > conf_thres
                ]

            # Filter by class
            if classes:
                x = x[(x[:, 5:6] == np.array(classes)).any(1)]

            # Apply finite constraint
            # if not torch.isfinite(x).all():
            #     x = x[torch.isfinite(x).all(1)]

            # If none remain process next image
            n = x.shape[0]  # number of boxes
            if not n:
                continue

            # Sort by confidence
            # x = x[x[:, 4].argsort(descending=True)]

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = nms(boxes, scores, iou_threshold=iou_thres)
            if len(i) > max_det:  # limit detections
                i = i[:max_det]
            if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
                try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                    iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                    weights = iou * scores[None]  # box weights
                    x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(
                        1, keepdim=True
                    )  # merged boxes
                    if redundant:
                        i = i[iou.sum(1) > 1]  # require redundancy
                except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                    print(x, i, x.shape, i.shape)
                    pass

            output[xi] = x[i]
            if (time.time() - t) > time_limit:
                break  # time limit exceeded

        return output

if __name__ == "__main__":
    model = YoloV8s()
    srcimg = cv2.imread("./imgs/te_safehat.jpg")
    '''
    for outrect in model(srcimg):
        x,y,w,h,cid,conf = int(outrect.rect.x), int(outrect.rect.y), int(outrect.rect.w), int(outrect.rect.h), int(outrect.label), round(float(outrect.prob) * 100)
        cv2.rectangle(srcimg, (x,y), (x+w, y+h), (0,255,255), 2)
        
        cv2.putText(srcimg, model.class_names[cid] + ":" + str(conf), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    '''
    tarimg = model(srcimg)

    cv2.imwrite("./imgs/ncnn_v8_safehat.jpg", tarimg)
