# openCV Face Recognition

## Introduce
> Face recognition system based on opencv


## File Description
- config.py (profile)
- human_info.py (Facial correspondence information)
- human.py
- main.py (Program Entry)
- training.py (Used for facial recognition training)


## How to use
1. Using the requirements.txt installation library
``` python
pip install -r requirements.txt

# If you are in China, you can use the following command
# pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
2. Prevent facial images from being displayed on the face_ In the data folder, name the secondary folder with label
for example：Amy's code is 2000, and her facial data path (face_data/2000)

1. Running training.py to generate recognition XM

2. Select the camera and start main.py
> The camera can use local or rtsp, rtmp protocols
``` python
h = HumanFace(camera=0)
# h = HumanFace(camera='rtsp://url')
# h = HumanFace(camera='rtmp://url')
```