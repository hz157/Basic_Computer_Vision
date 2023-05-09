# Object_Recognition_YOLOv8

> Using the object recognition code of yolov8, a certain performance is required to run the project, otherwise the recognition frame rate is only 1-2FPS

![Image](/docs/bccd57ce-ee40-11ed-ba2e-78af0827c486.png)

## Introduce
This fork recommends running on GPUs with NVIDIA CUDA cores

Extremely inefficient use of CPU

## How to use
1. Using the requirements.txt installation library
```
pip install -r requirements.txt

# If you are in China, you can use the following command
# pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

2. Select the camera and start main.py
> The camera can use local or rtsp, rtmp protocols
```
objectR = ObjectRecogize(camera=0)
objectR.Run()
h = HumanFace(camera=0)
# objectR = ObjectRecogize(camera='rtsp://url')
# objectR = ObjectRecogize(camera='rtmp://url')
```

3. Run ObjectRecogize.py