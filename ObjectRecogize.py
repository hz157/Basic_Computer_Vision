import cv2
import numpy as np
from PIL import Image
from ultralyticsplus import YOLO, render_result
import threading
import time

# load model
model = YOLO('ultralyticsplus/yolov8s')

# set model parameters
model.overrides['conf'] = 0.25  # NMS confidence threshold
model.overrides['iou'] = 0.45  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000  # maximum number of detections per image

receiveFrame = None
recoginzeFrame = None
rtmp = 0


class ObjectRecogize():
    receiveFrame = None
    recoginzeFrame = None
    camera = 0

    def __init__(self, camera=0):
        self.camera = camera
        self.frame_count = 0
        self.start_time = time.time()

    def Recognize(self):
        while True:
            if self.receiveFrame is not None:
                img = Image.fromarray(self.receiveFrame.astype(np.uint8))  # numpy.darray Convert PIL.Image
                results = model.predict(img)
                render = render_result(model=model, image=img, result=results[0])
                self.recoginzeFrame = np.array(render)
                self.frame_count += 1

    def Receive(self):
        print("start Receive")
        cap = cv2.VideoCapture(self.camera)
        while True:
            _, self.receiveFrame = cap.read()

    def Display(self):
        while True:
            if self.recoginzeFrame is not None:
                # Add frame rate information to the image
                end_time = time.time()
                elapsed_time = end_time - self.start_time
                fps = self.frame_count / elapsed_time
                fps_text = f"FPS: {fps:.2f}"
                cv2.putText(self.recoginzeFrame, fps_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("frame1", self.recoginzeFrame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    def Run(self):
        t1 = threading.Thread(target=self.Receive)
        t2 = threading.Thread(target=self.Recognize)
        t3 = threading.Thread(target=self.Display)
        t1.start()
        time.sleep(2)
        t2.start()
        time.sleep(2)
        t3.start()


if __name__ == '__main__':
    objectR = ObjectRecogize(camera=0)
    objectR.Run()
