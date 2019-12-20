# YOLO with TensorRT on Jetson Nano

Modified and customized version from [Jetson Nano: Deep Learning Inference Benchmarks Instructions](https://devtalk.nvidia.com/default/topic/1050377/jetson-nano/deep-learning-inference-benchmarking-instructions/).

## Setup

Install pre-requisites using: `$sh prebuild.sh`.

## trt-yolo-app

The trt-yolo-app located at `apps/trt-yolo` is a sample standalone app, which can be used to run inference on test images. This app does not have any deepstream dependencies and can be built independently. There is also an option of using custom build paths for TensorRT(-D TRT_SDK_ROOT)and OpenCV(-D OPENCV_ROOT). These are optional and not required if the libraries have already been installed.

```sh
cd apps/trt-yolo
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=Release ..
make
cd ../../../
./apps/trt-yolo/build/trt-yolo-app --flagfile=/path/to/config-file.txt
```

Refer to sample config files `yolov2.txt`, `yolov2-tiny.txt`, `yolov3.txt` and `yolov3-tiny.txt` in `config/` directory.

## GStreamer reference

[Accelerated GStreamer User Guide](https://developer.download.nvidia.cn/embedded/L4T/r32_Release_v1.0/Docs/Accelerated_GStreamer_User_Guide.pdf?77IO4det-OQLWN0hWfkVQCope_V7P4rDyYC4sgwfT2d0tU4dppX53NQBEX2irCqB4Gwuc8SKc-kWMOoX7qqrHLwTFDOdEECRE95Kbi39JQxwLw7bklgVoE4G5n5L6Y5y43tcaaYyVEPqFFpVt5l55D3NYeVrkLFd0ak3tqCsiut-BADOUjU)
