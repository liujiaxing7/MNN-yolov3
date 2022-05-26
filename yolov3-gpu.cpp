//
// Created by fandong on 2021/11/24.
//

#include "Interpreter.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include "ImageProcess.hpp"
#include <ctime>
#include <unistd.h>
#include "fstream"
#include <chrono>
#include <string.h>
#include <opencv2/highgui.hpp>

#define ERROR_PRINT(x) std::cout << "\033[31m" << (x) << "\033[0m" << std::endl

int class_nums = -1;
float prob_threshold = 0.25;
float nms_threshold = 0.45;
int boxes = 10647;

struct Object {
    cv::Rect_<float> rect;
    int label;
    float prob;
};

void read_classes(char *string);

static std::vector<std::string> class_names = {

};

static cv::Mat draw_objects(const cv::Mat &rgb, const std::vector<Object> &objects) {

    cv::Mat image = rgb.clone();
//    cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
    for (size_t i = 0; i < objects.size(); i++) {
        const Object &obj = objects[i];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label].c_str(), obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y),
                                      cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
    return image;
}

// sort vector of vectors by vectors' last value(bounding box confidence)
bool orderCriteria(std::vector<float> i, std::vector<float> j) { return (i.at(4) > j.at(4)); }

void ReadFile(std::string srcFile, std::vector<std::string> &image_files) {

    if (not access(srcFile.c_str(), 0) == 0) {
        ERROR_PRINT("no such File (" + srcFile + ")");
        return;
    }

    std::ifstream fin(srcFile.c_str());

    if (!fin.is_open()) {
        ERROR_PRINT("read file error (" + srcFile + ")");
        exit(0);
    }

    std::string s;
    while (getline(fin, s)) {
        image_files.push_back(s);
    }

    fin.close();
}

// intersection of union calculation
static float compute_iou(std::vector<float> vector1, std::vector<float> vector2) {
    float x10 = vector1.at(0);
    float y10 = vector1.at(1);
    float x20 = vector1.at(2);
    float y20 = vector1.at(3);
    float area0 = (x20 - x10) * (y20 - y10);

    float x11 = vector2.at(0);
    float y11 = vector2.at(1);
    float x21 = vector2.at(2);
    float y21 = vector2.at(3);
    float area1 = (x21 - x11) * (y21 - y11);

    float inner_x1 = x11 > x10 ? x11 : x10;
    float inner_y1 = y11 > y10 ? y11 : y10;
    float inner_x2 = x21 < x20 ? x21 : x20;
    float inner_y2 = y21 < y20 ? y21 : y20;
    float inner_area = (inner_x2 - inner_x1) * (inner_y2 - inner_y1);
    float iou = inner_area / (area0 + area1 - inner_area);

    return iou;
}

// function as numpy.clip
float clip(float n, float lower, float upper) {
    return std::max(lower, std::min(n, upper));
}

// compute angle between two lines formed by (x1,y1,x2,y2) and (x1,y1,x3,y3)
float compute_angle(float x1, float y1, float x2, float y2, float x3, float y3) {
    float dx21 = x2 - x1;
    float dx31 = x3 - x1;
    float dy21 = y2 - y1;
    float dy31 = y3 - y1;
    float m12 = sqrt(dx21 * dx21 + dy21 * dy21);
    float m13 = sqrt(dx31 * dx31 + dy31 * dy31);
    float theta = acos((dx21 * dx31 + dy21 * dy31) / (m12 * m13));
    return theta;
}

float CalculateDistance(std::vector<int> p1, std::vector<int> p2) {
    int SquareSum = 0;
    for (int i = 0; i < p1.size(); i++) {
        SquareSum += (p1[i] - p2[i]) * (p1[i] - p2[i]);
    }
    return sqrt(SquareSum);
}

std::vector<int> find_boundary_point(cv::Mat img, cv::Point ptCenter, cv::Point ptAngleMax) {
    cv::LineIterator it(img, ptCenter, ptAngleMax, 8);

    std::vector<int> pointBoundary(2, 0);

    // find a point detemining radius of vessel with center point
    std::vector<int> pointPrevious(3, 0);
    for (int idxIt = 0; idxIt < it.count; idxIt++, ++it) {
        std::vector<int> pointPresent(3, 0);
        for (int idxBgr = 0; idxBgr < 3; idxBgr++) {
            pointPresent.at(idxBgr) = int(img.at<cv::Vec3b>(it.pos())[idxBgr]);
        }
        float EucliDistance = CalculateDistance(pointPrevious, pointPresent);

        // display Euclidean distance between two neighbor points
        //if (idxIt >= 0) {
        //	std::cout << "distance: "<< EucliDistance <<std::endl;
        //}

        for (int idxBgr = 0; idxBgr < 3; idxBgr++) {
            pointPrevious.at(idxBgr) = int(img.at<cv::Vec3b>(it.pos())[idxBgr]);
        }

        if (idxIt > 0 && EucliDistance > 20) {
            pointBoundary.at(0) = it.pos().x;
            pointBoundary.at(1) = it.pos().y;
            break;
        }
    }

    return pointBoundary;
}

int main(int argc, char **argv) {

    if (argc < 3) {
        std::cout << "modelpath: mnnpath:\n"
                  << "data_path: images.txt\n"
                  << "classpath:: classes.txt" << std::endl;
        return -1;
    }
    auto start = std::chrono::steady_clock::now();

    const std::string mnn_path = argv[1];
    std::shared_ptr<MNN::Interpreter> my_interpreter = std::shared_ptr<MNN::Interpreter>(
            MNN::Interpreter::createFromFile(mnn_path.c_str()));

    // config
    MNN::ScheduleConfig config;
    int num_thread = 4;
    config.numThread = num_thread;
    MNN::BackendConfig backendConfig;
    backendConfig.precision = (MNN::BackendConfig::PrecisionMode) 2;
    config.backendConfig = &backendConfig;
    int forward = MNN_FORWARD_OPENCL;
    config.type = static_cast<MNNForwardType>(forward);


    // create session
    MNN::Session *my_session = my_interpreter->createSession(config);


    // session input pretreat
    MNN::Tensor *input_tensor = my_interpreter->getSessionInput(my_session, "input");
    my_interpreter->resizeTensor(input_tensor, {1, 3, 416, 416});

    std::string imagesTxt = argv[2];
    std::vector<std::string> imageNameList;
    std::vector<std::string> lidarNameList;

    read_classes(argv[3]);

    ReadFile(imagesTxt, imageNameList);
    const size_t size = imageNameList.size();
    for (size_t imgid = 0; imgid < size; ++imgid) {

        auto imageName = imageNameList.at(imgid);


        cv::Mat imgin = cv::imread(imageName);
        cv::Mat frame = cv::Mat(imgin.rows, imgin.cols, CV_8UC3, imgin.data);
        cv::Mat image;
        cv::resize(frame, image, cv::Size(416, 416), cv::INTER_LINEAR);

        // pass image to model(rgb format, pixels minus by 0 and then divided by 255.0)
        const float mean_vals[3] = {0.0, 0.0, 0.0};
        const float norm_vals[3] = {1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0};
        std::shared_ptr<MNN::CV::ImageProcess> pretreat(
                MNN::CV::ImageProcess::create(MNN::CV::RGB, MNN::CV::RGB, mean_vals, 3, norm_vals, 3));
        pretreat->convert(image.data, 416, 416, image.step[0], input_tensor);
        my_interpreter->runSession(my_session);

        // "boxes" -- [batch, num, 1, 4] ; "confs" -- [batch, num, num_classes]
        auto output_tensor_boxes = my_interpreter->getSessionOutput(my_session, "boxes");

        auto t_host_boxes = new MNN::Tensor(output_tensor_boxes, MNN::Tensor::CAFFE);
        output_tensor_boxes->copyToHostTensor(t_host_boxes);

        auto output_tensor_confs = my_interpreter->getSessionOutput(my_session, "confs");

        auto t_host_confs = new MNN::Tensor(output_tensor_confs, MNN::Tensor::CAFFE);
        output_tensor_confs->copyToHostTensor(t_host_confs);

        // convert arrays to vectors
        float *output_array_boxes = t_host_boxes->host<float>();
        float *output_array_confs = t_host_confs->host<float>();


        std::vector<float> output_vector_boxes{output_array_boxes, output_array_boxes + boxes * 4};
        std::vector<float> output_vector_confs{output_array_confs,
                                               output_array_confs + boxes * class_nums + class_nums};

        std::vector<std::vector<std::vector<float>>> vec(class_nums);
        std::cout << "vec.size(): " << vec.size() << std::endl;
        std::cout << std::endl;
        std::cout << "bounding boxes" << std::endl;

        // filter bounding boxes by threshold(0.4)
        for (int num = 0; num < boxes; num++) {
            std::vector<float>::const_iterator firstConfs = output_vector_confs.begin() + num * class_nums;
            std::vector<float>::const_iterator lastConfs = output_vector_confs.begin() + num * class_nums + class_nums;
            std::vector<float> prob_vector(firstConfs, lastConfs);
            int max_id = -1;
            float max_prob = -10000000000000000.0;
            for (int cls = 0; cls < class_nums; cls++) {
                if (prob_vector.at(cls) > max_prob) {
                    max_id = cls;
                    max_prob = prob_vector.at(cls);
                }
            }

            if (max_prob > prob_threshold) {
                std::vector<float>::const_iterator firstBoxes = output_vector_boxes.begin() + num * 4;
                std::vector<float>::const_iterator lastBoxes = output_vector_boxes.begin() + num * 4 + 4;
                std::vector<float> coord_vector(firstBoxes, lastBoxes);
                coord_vector.push_back(max_prob);

                for (int idx = 0; idx < 5; idx++) {
                    std::cout << coord_vector.at(idx) << " ";

                }
                std::cout << std::endl;
                vec.at(max_id).push_back(coord_vector);
            }
        }

        // nms
        for (int cls_nms = 0; cls_nms < class_nums; cls_nms++) {
            if (vec.at(cls_nms).size() == 0) {
                continue;
            } else {
                // sort by probability
                std::sort(vec.at(cls_nms).begin(), vec.at(cls_nms).end(), orderCriteria);

                int updated_size = vec.at(cls_nms).size();
                for (int i = 0; i < updated_size; i++) {
                    for (int j = i + 1; j < updated_size; j++) {
                        float score = compute_iou(vec.at(cls_nms).at(i), vec.at(cls_nms).at(j));
                        if (score >= nms_threshold) {
                            vec.at(cls_nms).erase(vec.at(cls_nms).begin() + j);
                            j = j - 1;
                            updated_size = vec.at(cls_nms).size();
                        }
                    }
                }
            }
        }

        std::vector<Object> objects;
        for (int cls_s = 0; cls_s < class_nums; cls_s++) {
            if (vec.at(cls_s).size() == 0) {
                continue;
            } else {
                for (int i = 0; i < vec.at(cls_s).size(); i++) {

                    Object obj;
                    obj.rect = cv::Rect_<float>(vec.at(cls_s).at(i).at(0) * imgin.cols,
                                                vec.at(cls_s).at(i).at(1) * imgin.rows,
                                                (vec.at(cls_s).at(i).at(2) - vec.at(cls_s).at(i).at(0)) * imgin.cols,
                                                (vec.at(cls_s).at(i).at(3) - vec.at(cls_s).at(i).at(1)) * imgin.rows);
                    obj.label = cls_s;
                    obj.prob = vec.at(cls_s).at(i).at(4);
                    objects.push_back(obj);

                }
            }
        }
//        auto imgshow = draw_objects(frame, objects);
//        cv::imshow("w", imgshow);
//        cv::waitKey(100);
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed1 = end - start1;
    std::cout << "data to gpu:" << elapsed1.count() << " s, ";
}

void read_classes(char *string) {
    std::fstream fin;
    fin.open(string, std::ios::in);
    std::string tmp;
    while (getline(fin, tmp)) {
        class_names.push_back(tmp);

    }
    class_nums = class_names.size();

}
