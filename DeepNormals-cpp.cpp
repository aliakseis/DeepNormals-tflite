// DeepNormals-cpp.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"


#include <iostream>
#include <tuple>
#include <string>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"


namespace {

using namespace cv;

// thinning stuff

enum ThinningTypes {
    THINNING_ZHANGSUEN = 0,  // Thinning technique of Zhang-Suen
    THINNING_GUOHALL = 1     // Thinning technique of Guo-Hall
};

// Applies a thinning iteration to a binary image
void thinningIteration(Mat img, int iter, int thinningType) {
    Mat marker = Mat::zeros(img.size(), CV_8UC1);

    if (thinningType == THINNING_ZHANGSUEN) {
        for (int i = 1; i < img.rows - 1; i++) {
            for (int j = 1; j < img.cols - 1; j++) {
                uchar p2 = img.at<uchar>(i - 1, j);
                uchar p3 = img.at<uchar>(i - 1, j + 1);
                uchar p4 = img.at<uchar>(i, j + 1);
                uchar p5 = img.at<uchar>(i + 1, j + 1);
                uchar p6 = img.at<uchar>(i + 1, j);
                uchar p7 = img.at<uchar>(i + 1, j - 1);
                uchar p8 = img.at<uchar>(i, j - 1);
                uchar p9 = img.at<uchar>(i - 1, j - 1);

                int A = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) + (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
                        (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) + (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
                int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
                int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
                int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

                if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0) marker.at<uchar>(i, j) = 1;
            }
        }
    }
    if (thinningType == THINNING_GUOHALL) {
        for (int i = 1; i < img.rows - 1; i++) {
            for (int j = 1; j < img.cols - 1; j++) {
                uchar p2 = img.at<uchar>(i - 1, j);
                uchar p3 = img.at<uchar>(i - 1, j + 1);
                uchar p4 = img.at<uchar>(i, j + 1);
                uchar p5 = img.at<uchar>(i + 1, j + 1);
                uchar p6 = img.at<uchar>(i + 1, j);
                uchar p7 = img.at<uchar>(i + 1, j - 1);
                uchar p8 = img.at<uchar>(i, j - 1);
                uchar p9 = img.at<uchar>(i - 1, j - 1);

                int C = ((!p2) & (p3 | p4)) + ((!p4) & (p5 | p6)) + ((!p6) & (p7 | p8)) + ((!p8) & (p9 | p2));
                int N1 = (p9 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8);
                int N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p9);
                int N = N1 < N2 ? N1 : N2;
                int m = iter == 0 ? ((p6 | p7 | (!p9)) & p8) : ((p2 | p3 | (!p5)) & p4);

                if ((C == 1) && ((N >= 2) && ((N <= 3)) & (m == 0))) marker.at<uchar>(i, j) = 1;
            }
        }
    }

    img &= ~marker;
}

// Apply the thinning procedure to a given image
void thinning(InputArray input, OutputArray output, int thinningType = THINNING_ZHANGSUEN) {
    Mat processed = input.getMat().clone();
    // Enforce the range of the input image to be in between 0 - 255
    processed /= 255;

    Mat prev = Mat::zeros(processed.size(), CV_8UC1);
    Mat diff;

    do {
        thinningIteration(processed, 0, thinningType);
        thinningIteration(processed, 1, thinningType);
        absdiff(processed, prev, diff);
        processed.copyTo(prev);
    } while (countNonZero(diff) > 0);

    processed *= 255;

    output.assign(processed);
}

} // namespace


namespace {

auto load_linedrawing(const char* Path) {
    auto img = cv::imread(Path, cv::IMREAD_GRAYSCALE);
    cv::bitwise_not(img, img); //invert image
    cv::Mat thresh1;
    cv::threshold(img, thresh1, 24, 255, cv::THRESH_BINARY);
    return thresh1;
}

auto PrepareMultiScale(const cv::Mat& src) {
    cv::Mat img;
    src.convertTo(img, CV_32F);

    enum { size = 256 };
    cv::Mat img_pad = cv::Mat::zeros(img.rows + 2 * size, img.cols + 2 * size, CV_32FC1);
    img.copyTo(img_pad(cv::Rect(size + 1, size + 1, img.cols, img.rows)));

    //resized version of image for global view
    cv::Mat img_2tmp;
    cv::resize(img, img_2tmp, {}, 0.5, 0.5, cv::INTER_LINEAR);
    cv::Mat img_2 = cv::Mat::zeros(img_2tmp.rows + 2 * size, img_2tmp.cols + 2 * size, CV_32FC1);
    img_2tmp.copyTo(img_2(cv::Rect(size + 1, size + 1, img_2tmp.cols, img_2tmp.rows)));

    cv::Mat img_4tmp;
    cv::resize(img_2tmp, img_4tmp, {}, 0.5, 0.5, cv::INTER_LINEAR);
    cv::Mat img_4 = cv::Mat::zeros(img_4tmp.rows + 2 * size, img_4tmp.cols + 2 * size, CV_32FC1);
    img_4tmp.copyTo(img_4(cv::Rect(size + 1, size + 1, img_4tmp.cols, img_4tmp.rows)));

    return std::make_tuple(img_pad, img_2, img_4);
}


auto BorderHandle(int x, int size_2, int lenn) {

    int xm, Xm;

    if ((x - size_2) < 0) {
        xm = 0;
        Xm = size_2 - x;
    }
    else {
        xm = (x - size_2);
        Xm = 0;
    }

    int xM, XM;

    if ((x + size_2) > lenn) {
        xM = lenn;
        XM = size_2 + (lenn - x);
    }
    else {
        xM = x + size_2;
        XM = 2 * size_2;
    }
    return std::make_tuple(xm, xM, Xm, XM);
}


auto CropMultiScale_ZeroPadding_2(int x, int y, const cv::Mat& image, const cv::Mat& image_2, const cv::Mat& image_4, int size) {

    std::vector<cv::Mat> img_blank(3);
    for (auto& v : img_blank)
        v = cv::Mat::zeros(size, size, CV_32FC1);

    auto x1 = int(x / 2) + size + 1;
    auto y1 = int(y / 2) + size + 1;
    auto x2 = int(x / 4) + size + 1;
    auto y2 = int(y / 4) + size + 1;
    x = x + size + 1;
    y = y + size + 1;
    size = int(size / 2);

    {
        auto[xm, xM, Xm, XM] = BorderHandle(x1, size, image_2.cols);
        auto[ym, yM, Ym, YM] = BorderHandle(y1, size, image_2.rows);
        image_2({ Point(xm, ym), Point(xM, yM) }).copyTo(img_blank[1]({ Point(Xm, Ym), Point(XM, YM) }));
    }

    {
        auto[xm, xM, Xm, XM] = BorderHandle(x2, size, image_4.cols);
        auto[ym, yM, Ym, YM] = BorderHandle(y2, size, image_4.rows);
        image_4({ Point(xm, ym), Point(xM, yM) }).copyTo(img_blank[2]({ Point(Xm, Ym), Point(XM, YM) }));
    }

    {
        auto[xm, xM, Xm, XM] = BorderHandle(x, size, image.cols);
        auto[ym, yM, Ym, YM] = BorderHandle(y, size, image.rows);
        image({ Point(xm, ym), Point(xM, yM) }).copyTo(img_blank[0]({ Point(Xm, Ym), Point(XM, YM) }));
    }

    for (auto& v : img_blank)
    {
        v /= 127.5;
        v -= 1.0;

    }

    cv::Mat result;
    cv::merge(img_blank, result);
    
    return result;
}

} // namespace


int main(int argc, char** argv)
{
    if (argc < 3)
        return 1;

    try {

        auto img = load_linedrawing(argv[1]);

        //Load Mask
        auto Mask = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);


           // create model
        std::unique_ptr<tflite::FlatBufferModel> model =
            tflite::FlatBufferModel::BuildFromFile("/solutions/DeepNormals-tflite/model.tflite");
        tflite::ops::builtin::BuiltinOpResolver resolver;
        std::unique_ptr<tflite::Interpreter> interpreter;
        tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);
        interpreter->AllocateTensors();

        // get input & output layer
        float* inputLayer = interpreter->typed_input_tensor<float>(0);
        float* outputLayer = interpreter->typed_output_tensor<float>(0);


        thinning(img, img);

        for (int y = 0; y < Mask.rows; ++y)
            for (int x = 0; x < Mask.cols; ++x)
                if (Mask.at<uchar>(y, x))
                {
                    auto& v = img.at<uchar>(y, x);
                    if (v != 255)
                        v = 160;
                }

        //cv::imshow("img", img);
        //cv::waitKey();

        auto[img_pad, img_2, img_4] = PrepareMultiScale(img);

        //////////////////////////////////////////////////////////////////////////

        int height = img.rows;
        int width = img.cols;
        enum { size = 256 };

        const auto nb_grids = 40;

        auto ind = 0;

        cv::Mat recfin = cv::Mat::zeros(height + 600, width + 600, CV_32FC3);

        for (int offset = 0; offset < 256; offset += int(256 / nb_grids)) {
            std::vector<cv::Mat> subBatch;
            std::vector<cv::Point> pos;
            auto index = 0;

            for (int j = 0; j < int(height / 256) + 2; ++j) {
                int y = j * 256 + offset - 128;
                for (int i = 0; i <int(width / 256) + 2; ++i) {
                    int x = i * 256 + offset - 128;
                    try {
                        auto sub = CropMultiScale_ZeroPadding_2(x, y, img_pad, img_2, img_4, size);

                        //cv::imshow("sub", sub);
                        //cv::waitKey();

                        subBatch.push_back(sub);
                        ++index;
                        pos.push_back({ x, y });
                    }
                    catch (...) {
                        throw;
                    }
                }
            }

            cv::Mat rec = cv::Mat::zeros(height + 900, width + 900, CV_32FC3);

            const int off = 260;
            const int s = int(size / 2);

            for (int i = 0; i < subBatch.size(); ++i)
            {
                //cv::imwrite("C:/solutions/DeepNormals/saved_cpp//input_" + std::to_string(ind + i) + ".png", subBatch[i] * 127.5 + 127.5, { IMWRITE_PNG_COMPRESSION, 9 });

                enum { WIDTH  = size, HEIGHT = size, CHANNEL = 3 };
                // flatten rgb image to input layer.
                float* inputImg_ptr = subBatch[i].ptr<float>(0);
                memcpy(inputLayer, inputImg_ptr,
                    WIDTH * HEIGHT * CHANNEL * sizeof(float));

                // compute model instance
                interpreter->Invoke();

                cv::Mat pred(size, size, CV_32FC3, outputLayer);


                //cv::imwrite("C:/solutions/DeepNormals/saved_cpp//output_" + std::to_string(ind + i) + ".png", pred * 127.5 + 127.5, { IMWRITE_PNG_COMPRESSION, 9 });

                int x = off + pos[i].x;
                int y = off + pos[i].y;
                try {
                    rec({ Point(x - s, y - s), Point(x + s, y + s) }) += pred;
                }
                catch (...) {
                    throw;
                }
            }

            ++ind;

            recfin({ 0, 0, width, height }) += rec({ 260, 260, width, height });
        }

        //std::cout << ind << '\n';

        std::vector<cv::Mat> img_blank;
        cv::split(recfin, img_blank);
        for (auto& v : img_blank)
        {
            v *= (.5 / ind);
            v += .5;

        }
        cv::merge(img_blank, recfin);

        cv::Mat result = recfin({ 0, 0, width, height });

        cv::imshow("result", result);
        cv::waitKey();

    }
    catch (const std::exception& ex) {
        std::cerr << typeid(ex).name() << ": " << ex.what() << '\n';
    }
}
