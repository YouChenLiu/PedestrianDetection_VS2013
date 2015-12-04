#include <opencv2/highgui.hpp>
#include "../CommonClassLibrary/myFeatureExtractor/myFeatureExtractor.h"

int main(void) {
    cv::Mat mImg = cv::Mat::zeros(64, 64, CV_8UC1);
    myFeatureExtractor e(mImg, cv::Size2i(8, 8));
    e.EnableFeature(myFeatureExtractor::Features::HOG_WITHOUT_NORM);
    e.EnableFeature(myFeatureExtractor::Features::LBP_8_1_UNIFORM);
    std::vector<float> vfFeature;
    e.Describe(0, 0, vfFeature);
    return 0;
}