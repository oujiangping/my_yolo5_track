import os
from time import time

import cv2
import numpy as np
import torch
from deep_sort_pytorch.deep_sort import DeepSort
from deep_sort_pytorch.utils.parser import get_config

os.environ['http_proxy'] = 'http://127.0.0.1:41091'
os.environ['https_proxy'] = 'https://127.0.0.1:41091'

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5n6', pretrained=True, device='cpu')
names = model.names


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def track_and_plot(results, frame):
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    boxes = []
    confs = []
    clss = []
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.2:
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
            boxes.append([x1, y1, x2, y2])
            confs.append(row[4])
            clss.append(labels[i])
    if len(clss):
        xywhs = torch.from_numpy(xyxy2xywh(np.array(boxes)))
        confs = torch.from_numpy(np.array(confs))
        clss = torch.from_numpy(np.array(clss))

        outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), frame)

        if len(outputs) > 0:
            for j, (output, conf) in enumerate(zip(outputs, confs)):
                bboxes = output[0:4]
                id = output[4]
                cls = output[5]

                c = int(cls)  # integer class
                label = f'{id} {names[c]} {conf:.2f}'
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (bboxes[0], bboxes[1]), (bboxes[2], bboxes[3]), bgr, 2)
                cv2.putText(frame, label, (bboxes[0], bboxes[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
    return frame


def score_frame(frame):
    frames = [frame]
    results = model(frames)
    labels, cord = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
    return labels, cord


max_cosine_distance = 0.3
nms_max_overlap = 1.0
nn_budget = None

cfg = get_config()
cfg.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)

player = cv2.VideoCapture(0)
while True:
    start_time = time()
    ret, frame = player.read()
    assert ret
    print("got a frame")
    results = score_frame(frame)

    frame = track_and_plot(results, frame)
    cv2.imshow("video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    end_time = time()
    fps = 1 / np.round(end_time - start_time, 3)
    print(f"Frames Per Second : {fps}")
player.release()
cv2.destroyAllWindows()
