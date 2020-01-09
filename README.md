# Video YOLO with TensorRT on Jetson Nano

Modified and customized version of [Jetson Nano: Deep Learning Inference Benchmarks Instructions](https://devtalk.nvidia.com/default/topic/1050377/jetson-nano/deep-learning-inference-benchmarking-instructions/).

Run real-time object detections on Jetson Nano with TensorRT optimized YOLO network.

## Performance

|   Network   | Framerate |
| :---------: | :-------: |
|   YOLOv3    |  2 to 5   |
| YOLOv3-tiny |    24     |

YOLOv3-tiny is way faster but yields poor detection results.

## Setup

Install prerequisites and fetch weights:

```bash
chmod +x prebuild.sh
./prebuild.sh
```

## Build & Run

```bash
cd src
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=Release ..
make
cd ../../
./src/build/trt-yolo-app --flagfile=/path/to/config-file.txt

# e.g.
./src/build/trt-yolo-app --flagfile=config/yolov3-tiny.txt
```

Refer to sample config files `yolov2.txt`, `yolov2-tiny.txt`, `yolov3.txt` and `yolov3-tiny.txt` in `config/` directory.

## GStreamer Reference

This project uses `GStreamer` to utilize Nvidia's hardware acceleration on video capture, encoding and decoding. You should change some code in `src/main.cpp` to reflect the camera setup you are using for Jetson Nano.

For example, to test I have the following run on Jetson Nano:

```bash
gst-launch-1.0 nvarguscamerasrc ! \
    'video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, format=(string)NV12, framerate=(fraction)30/1' ! \
    nvvidconv flip-method=0 ! 'video/x-raw, format=(string)BGRx' ! \
    videoconvert ! 'video/x-raw, format=(string)BGR' ! \
    videoconvert ! 'video/x-raw, format=(string)RGB' ! \
    videoconvert ! nvvidconv ! 'video/x-raw(memory:NVMM), format=(string)NV12' ! \
    nvv4l2h265enc insert-sps-pps=true ! 'video/x-h265, stream-format=(string)byte-stream' ! \
    queue ! h265parse ! queue ! \
    rtph265pay ! queue ! \
    udpsink host=192.168.1.194 port=1234
```

The command above read frames from the camera and then send the stream to `192.168.1.194` which is my desktop address in LAN.

From my desktop, run:

```bash
gst-launch-1.0 udpsrc port=1234 ! \
    application/x-rtp,encoding-name=H265 ! queue ! \
    rtph265depay ! queue ! avdec_h265 ! queue ! autovideosink
```

There you can see the streaming video on your desktop, which is being captured on Jetson Nano.

This means the GStreamer pipeline is valid, so you could use those commands in OpenCV `VideoCapture`'s `src` and `VideoWriter`'s `des`. See `src/main.cpp` for details.

For more on `GStreamer` and `nvarguscamerasrc`, see [Accelerated GStreamer User Guide](https://developer.download.nvidia.cn/embedded/L4T/r32_Release_v1.0/Docs/Accelerated_GStreamer_User_Guide.pdf?77IO4det-OQLWN0hWfkVQCope_V7P4rDyYC4sgwfT2d0tU4dppX53NQBEX2irCqB4Gwuc8SKc-kWMOoX7qqrHLwTFDOdEECRE95Kbi39JQxwLw7bklgVoE4G5n5L6Y5y43tcaaYyVEPqFFpVt5l55D3NYeVrkLFd0ak3tqCsiut-BADOUjU).
