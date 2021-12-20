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

stater = {}
counter = {"up": 0, "down": 0, "left": 0, "right": 0}


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
    y_line = y_shape / 2
    x_line = x_shape / 2
    boxes = []
    confs = []
    clss = []
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.1 and int(labels[i]) == 0:
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
            boxes.append([x1, y1, x2, y2])
            confs.append(row[4])
            clss.append(labels[i])
    if len(clss):
        xywhs = torch.from_numpy(xyxy2xywh(np.array(boxes)))
        confs = torch.from_numpy(np.array(confs))
        clss = torch.from_numpy(np.array(clss))

        outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), frame, use_yolo_preds=True)

        person_count_now = 0
        if len(outputs) > 0:
            for j, (output, conf) in enumerate(zip(outputs, confs)):
                bboxes = output[0:4]
                id = output[4]
                cls = output[5]

                if int(cls) == 0:
                    person_count_now += 1

                center_y = int((bboxes[1] + bboxes[3]) / 2)
                center_x = int((bboxes[0] + bboxes[2]) / 2)
                if int(cls) == 0:
                    if str(id) not in stater.keys():
                        stater[str(id)] = {
                            "now_y": center_y,
                            "old_y": center_y,
                            "now_x": center_x,
                            "old_x": center_x,
                            "down": False,
                            "up": False,
                            "left": False,
                            "right": False
                        }
                    else:
                        stater[str(id)]["old_y"] = stater[str(id)]["now_y"]
                        stater[str(id)]["now_y"] = center_y
                        stater[str(id)]["old_x"] = stater[str(id)]["now_x"]
                        stater[str(id)]["now_x"] = center_x
                        if stater[str(id)]["old_y"] <= int(y_line) < stater[str(id)]["now_y"] and \
                                (not stater[str(id)]["down"]) and int(cls) == 0:
                            counter["down"] += 1
                            stater[str(id)]["down"] = True
                        if stater[str(id)]["old_y"] >= int(y_line) > stater[str(id)]["now_y"] and \
                                (not stater[str(id)]["up"]) and int(cls) == 0:
                            counter["up"] += 1
                            stater[str(id)]["up"] = True

                        if stater[str(id)]["old_x"] <= int(x_line) < stater[str(id)]["now_x"] and \
                                (not stater[str(id)]["right"]) and int(cls) == 0:
                            counter["right"] += 1
                            stater[str(id)]["right"] = True
                        if stater[str(id)]["old_x"] >= int(x_line) > stater[str(id)]["now_x"] and \
                                (not stater[str(id)]["left"]) and int(cls) == 0:
                            counter["left"] += 1
                            stater[str(id)]["left"] = True
                c = int(cls)
                label = f'{id} {names[c]} {conf:.2f}'
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (bboxes[0], bboxes[1]), (bboxes[2], bboxes[3]), bgr, 2)
                cv2.putText(frame, label, (bboxes[0], bboxes[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
                cv2.circle(frame, (int((bboxes[0] + bboxes[2]) / 2), int((bboxes[1] + bboxes[3]) / 2)), 5, (0, 0, 255),
                           10)
    cv2.line(frame, (0, int(y_line)), (int(x_shape), int(y_line)), (255, 0, 0), 5)
    cv2.line(frame, (int(x_line), 0), (int(x_line), int(y_shape)), (255, 0, 0), 5)
    cv2.putText(frame, "person up:" + str(counter["up"]), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255),
                2)
    cv2.putText(frame, "person down:" + str(counter["down"]), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 0, 255), 2)
    cv2.putText(frame, "person left:" + str(counter["left"]), (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255),
                2)
    cv2.putText(frame, "person right:" + str(counter["right"]), (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 0, 255), 2)
    cv2.putText(frame, "person now:" + str(person_count_now), (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 0, 255), 2)

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
                    use_cuda=False)

player = cv2.VideoCapture("/Users/oujiangping/Downloads/test1.mp4")
#player = cv2.VideoCapture(0)
frame_count = 1
while True:
    start_time = time()
    ret, frame = player.read()
    assert ret
    frame_count += 1
    if frame_count % 2 == 0:
        continue
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
