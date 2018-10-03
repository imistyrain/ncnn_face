//MobileNet-SSD Face detection demo
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <iostream>

#include "net.h"
#include "mrdir.h"
#include "mropencv.h"
#pragma comment(lib,"ncnn.lib")
#define USE_PARAM_BIN 0
#if USE_PARAM_BIN
#include "mobilenet_ssd_voc_ncnn.id.h"
#endif
struct Object{
    cv::Rect rec;
    int class_id;
    float prob;
};
int input_size = 300;
const char* class_names[] = {"background","face"};
std::string modelnameprefix = "../ncnn_face";
static int detect_mobilenet(cv::Mat& raw_img, float show_threshold)
{
    ncnn::Net mobilenet;
    /*
     * model is  converted from https://github.com/chuanqi305/MobileNet-SSD
     * and can be downloaded from https://drive.google.com/open?id=0ByaKLD9QaPtucWk0Y0dha1VVY0U
     */
    int img_h = raw_img.size().height;
    int img_w = raw_img.size().width;
#if USE_PARAM_BIN
    mobilenet.load_param_bin((modelnameprefix + ".param.bin").c_str());
#else
	mobilenet.load_param((modelnameprefix+".param").c_str());
#endif
	mobilenet.load_model((modelnameprefix + ".bin").c_str());
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(raw_img.data, ncnn::Mat::PIXEL_BGR, raw_img.cols, raw_img.rows, input_size, input_size);

    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1.0/127.5,1.0/127.5,1.0/127.5};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Mat out;

    ncnn::Extractor ex = mobilenet.create_extractor();
    ex.set_light_mode(true);
    //ex.set_num_threads(4);
	cv::TickMeter tm;
	tm.start();
#if USE_PARAM_BIN
    ex.input(mobilenet_ssd_voc_ncnn_param_id::BLOB_data, in);
    ex.extract(mobilenet_ssd_voc_ncnn_param_id::BLOB_detection_out, out);    
#else
    ex.input("data", in);
    ex.extract("detection_out", out);
#endif
	tm.stop();
	std::cout << tm.getTimeMilli() << "ms" << std::endl;
    printf("%d %d %d\n", out.w, out.h, out.c);
    std::vector<Object> objects;
    for (int iw=0;iw<out.h;iw++)
    {
        Object object;
        const float *values = out.row(iw);
        object.class_id = values[0];
        object.prob = values[1];
        object.rec.x = values[2] * img_w;
        object.rec.y = values[3] * img_h;
        object.rec.width = values[4] * img_w - object.rec.x;
        object.rec.height = values[5] * img_h - object.rec.y;
        objects.push_back(object);
    }

    for(int i = 0;i<objects.size();++i)
    {
        Object object = objects.at(i);
        if(object.prob > show_threshold)
        {
            cv::rectangle(raw_img, object.rec, cv::Scalar(255, 0, 0));
            std::ostringstream pro_str;
            pro_str<<object.prob;
            std::string label = std::string(class_names[object.class_id]) + ": " + pro_str.str();
            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            cv::rectangle(raw_img, cv::Rect(cv::Point(object.rec.x, object.rec.y- label_size.height),
                                  cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), CV_FILLED);
            cv::putText(raw_img, label, cv::Point(object.rec.x, object.rec.y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        }
    }
    cv::imshow("result",raw_img);
    cv::waitKey();

    return 0;
}

int test_img(const char* imagepath = "../images/test.jpg")
{
    cv::Mat m = cv::imread(imagepath, CV_LOAD_IMAGE_COLOR);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    detect_mobilenet(m,0.5);

    return 0;
}

class MobileNetFaceDetector
{
public:
    MobileNetFaceDetector(const std::string modelnameprefix = "../ncnn_face")
    {
#if USE_PARAM_BIN
        mobilenet.load_param_bin((modelnameprefix + ".param.bin").c_str());
#else
        mobilenet.load_param((modelnameprefix + ".param").c_str());
#endif
        mobilenet.load_model((modelnameprefix + ".bin").c_str());
    }
    ncnn::Mat detect(const cv::Mat raw_img);
    cv::Mat drawResult(cv::Mat raw_img, const ncnn::Mat out, float show_threshold = 0.5);
private:
    ncnn::Net mobilenet;
};

ncnn::Mat MobileNetFaceDetector::detect(const cv::Mat raw_img)
{
    int img_h = raw_img.size().height;
    int img_w = raw_img.size().width;
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(raw_img.data, ncnn::Mat::PIXEL_BGR, raw_img.cols, raw_img.rows, input_size, input_size);
    const float mean_vals[3] = { 127.5f, 127.5f, 127.5f };
    const float norm_vals[3] = { 1.0 / 127.5,1.0 / 127.5,1.0 / 127.5 };
    in.substract_mean_normalize(mean_vals, norm_vals);
    ncnn::Mat out;
    ncnn::Extractor ex = mobilenet.create_extractor();
    ex.set_light_mode(true);
    //ex.set_num_threads(4);
    cv::TickMeter tm;
    tm.start();
#if USE_PARAM_BIN
    ex.input(mobilenet_ssd_voc_ncnn_param_id::BLOB_data, in);
    ex.extract(mobilenet_ssd_voc_ncnn_param_id::BLOB_detection_out, out);
#else
    ex.input("data", in);
    ex.extract("detection_out", out);
#endif
    tm.stop();
    std::cout << tm.getTimeMilli() << "ms" << std::endl;
    //printf("%d %d %d\n", out.w, out.h, out.c);
    return out;
}

cv::Mat MobileNetFaceDetector::drawResult(cv::Mat raw_img, const ncnn::Mat out, float show_threshold)
{
    int img_h = raw_img.size().height;
    int img_w = raw_img.size().width;
    std::vector<Object> objects;
    for (int iw = 0; iw<out.h; iw++)
    {
        Object object;
        const float *values = out.row(iw);
        object.class_id = values[0];
        object.prob = values[1];
        object.rec.x = values[2] * img_w;
        object.rec.y = values[3] * img_h;
        object.rec.width = values[4] * img_w - object.rec.x;
        object.rec.height = values[5] * img_h - object.rec.y;
        objects.push_back(object);
    }

    for (int i = 0; i<objects.size(); ++i)
    {
        Object object = objects.at(i);
        if (object.prob > show_threshold)
        {
            cv::rectangle(raw_img, object.rec, cv::Scalar(255, 0, 0));
            std::ostringstream pro_str;
            pro_str << object.prob;
            std::string label = std::string(class_names[object.class_id]) + ": " + pro_str.str();
            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            cv::rectangle(raw_img, cv::Rect(cv::Point(object.rec.x, object.rec.y - label_size.height),
                cv::Size(label_size.width, label_size.height + baseLine)),
                cv::Scalar(255, 255, 255), CV_FILLED);
            cv::putText(raw_img, label, cv::Point(object.rec.x, object.rec.y),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        }
    }
    return raw_img;
}

int test_camera(float show_threshold=0.5)
{
    cv::VideoCapture capture(0);
    MobileNetFaceDetector md;
    cv::Mat raw_img;
    while (true)
    {
        capture >> raw_img;
        if (!raw_img.data)
        {
            break;
        }
        auto result = md.detect(raw_img);
        auto show = md.drawResult(raw_img, result);
        cv::imshow("img", show);
        cv::waitKey(1);
    }
    return 0;
}

int test_dir(const std::string dir="../images")
{
    MobileNetFaceDetector md;
    auto files = getAllFilesinDir(dir);
    for (auto file:files)
    {
        std::string filepath = dir + "/" + file;
        cv::Mat raw_img = cv::imread(filepath);
        auto result = md.detect(raw_img);
        auto show = md.drawResult(raw_img, result);
        cv::imshow("img", show);
        //cv::imwrite("result.jpg", show);
        cv::waitKey();
    }
    return 0;
}

int main(int argc, char** argv)
{
    //test_img();
    test_camera();
    //test_dir();
    return 0;
}