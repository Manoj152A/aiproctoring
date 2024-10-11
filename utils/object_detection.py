from ultralytics import YOLO

class ObjectDetection:
    def __init__(self):
        self.model = YOLO('yolov5su.pt')  # Make sure this file is in your project directory

    def detect(self, img):
        results = self.model(img)
        return [
            {
                'class': int(det[5]),
                'confidence': float(det[4]),
                'bbox': [float(x) for x in det[:4]]
            }
            for det in results[0].boxes.data.tolist()
        ]