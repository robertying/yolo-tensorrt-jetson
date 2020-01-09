/**
MIT License

Copyright (c) 2018 NVIDIA CORPORATION. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*
*/

#include "ds_image.h"
#include "trt_utils.h"
#include "yolo_config_parser.h"
#include "yolov3.h"

#include <cstdio>
#include <csignal>
#include <cstdlib>
#include <unistd.h>
#include <queue>
#include <thread>
#include <sys/time.h>
#include <opencv2/videoio.hpp>

std::queue<cv::Mat> readQueue;
std::queue<cv::Mat> writeQueue;

volatile sig_atomic_t stop = 0;

void sigint_handler(int s){
    printf("\nCleaning resources...\n");
    stop = 1;
}

void readFrame(cv::VideoCapture& cap) {
    cv::Mat frame;

    while (!stop)
    {
        cap >> frame;

        if (frame.empty()) {
            std::cout << "Could not read camera" << std::endl;
            return;
        }

        readQueue.push(frame);
    }
}

void processFrame(std::unique_ptr<Yolo>& inferNet) {
    DsImage dsImage;
    while (!stop)
    {
        if (readQueue.empty()){
            continue;
        }

        cv::Mat frame = readQueue.front();
        readQueue.pop();


        // Load a new batch
        dsImage = DsImage(frame, inferNet->getInputH(), inferNet->getInputW());
        cv::Mat trtInput = blobFromDsImage(dsImage, inferNet->getInputH(), inferNet->getInputW());

        // struct timeval inferStart, inferEnd;
        // gettimeofday(&inferStart, NULL);

        inferNet->doInference(trtInput.data, 1);

        // gettimeofday(&inferEnd, NULL);
        // double inferElapsed = ((inferEnd.tv_sec - inferStart.tv_sec)
        //                 + (inferEnd.tv_usec - inferStart.tv_usec) / 1000000.0)
        //                 * 1000;
        // std::cout << "Frame process time: " << inferElapsed << "ms" << std::endl;

        auto binfo = inferNet->decodeDetections(0, dsImage.getImageHeight(),
                                                dsImage.getImageWidth());
        auto remaining
            = nmsAllClasses(inferNet->getNMSThresh(), binfo, inferNet->getNumClasses());
        for (auto b : remaining)
        {
            printPredictions(b, inferNet->getClassName(b.label));
            dsImage.addBBox(b, inferNet->getClassName(b.label));
        }

        cv::Mat img = dsImage.getMaskedImage();

        writeQueue.push(img);
    }
}

void writeFrame(cv::VideoWriter& out) {
    while (!stop) {
        if (writeQueue.empty()) {
            continue;
        }

        cv::Mat img = writeQueue.front();
        writeQueue.pop();

        // struct timeval inferStart, inferEnd;
        // gettimeofday(&inferStart, NULL);

        out << img;

        // gettimeofday(&inferEnd, NULL);
        // double inferElapsed = ((inferEnd.tv_sec - inferStart.tv_sec)
        //                  + (inferEnd.tv_usec - inferStart.tv_usec) / 1000000.0)
        //                  * 1000;
        // std::cout << "Frame write time: " << inferElapsed << "ms" << std::endl;
    }
}

int main(int argc, char** argv)
{
    // Flag set in the command line overrides the value in the flagfile
    gflags::SetUsageMessage(
        "Usage : trt-yolo-app --flagfile=</path/to/config_file.txt> --<flag>=value ...");

    // parse config params
    yoloConfigParserInit(argc, argv);
    NetworkInfo yoloInfo = getYoloNetworkInfo();
    InferParams yoloInferParams = getYoloInferParams();
    uint batchSize = getBatchSize();

    std::unique_ptr<Yolo> inferNet{nullptr};
    inferNet = std::unique_ptr<Yolo>{new YoloV3(batchSize, yoloInfo, yoloInferParams)};

    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = sigint_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);

    char* cameraSrc = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
    cv::VideoCapture cap(cameraSrc);
    if (!cap.isOpened())
    {
        std::cout  << "Could not open camera" << std::endl;
        return -1;
    }

    int img_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int img_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double frame_rate = cap.get(cv::CAP_PROP_FPS);
    printf("Video spec: %dx%d@%dfps\n", img_width, img_height, int(frame_rate));

    char* fileOutput = "appsrc ! videoconvert ! nvvidconv ! video/x-raw(memory:NVMM), format=(string)NV12 ! nvv4l2h265enc ! h265parse ! qtmux ! filesink location=output.mp4 ";
    char* tcpOutput = "appsrc ! videoconvert ! nvvidconv ! video/x-raw(memory:NVMM), format=(string)NV12 ! nvv4l2h265enc insert-sps-pps=true ! queue ! h265parse ! queue ! rtph265pay mtu=65507 ! queue ! udpsink host=192.168.1.194 port=1234";
    cv::VideoWriter out(fileOutput, 0, 24.0, cv::Size(1920, 1080), true);
    if (!out.isOpened())
    {
        std::cout  << "Could not open write file"<< std::endl;
        return -1;
    }

    std::thread read(readFrame, std::ref(cap));
    std::thread process(processFrame, std::ref(inferNet));
    std::thread write(writeFrame, std::ref(out));

    read.join();
    process.join();
    write.join();

    return 0;
}
