import torch
import os

os.environ['http_proxy'] = 'http://127.0.0.1:41091'
os.environ['https_proxy'] = 'https://127.0.0.1:41091'

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x6', pretrained=True)

# Images
imgs = ['/Users/oujiangping/Downloads/person.jpeg']  # batch of images

# Inference
results = model(imgs)

# Results
results.print()
results.show()  # or .show()

results.xyxy[0]  # img1 predictions (tensor)
results.pandas().xyxy[0]  # img1 predictions (pandas)