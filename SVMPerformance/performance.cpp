#include <iostream>
#include "../CommonClassLibrary/myImageSequence/myImageSequence.h"
#include "../CommonClassLibrary/myFeatureExtractor/myFeatureExtractor.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <fstream>
#include <time.h>

int main(void) {
    const cv::Size2i BlockSize(8, 8);
    int iTimes = 0;
    int iDuration = 0;

    //  load the smv model
    cv::Ptr<cv::ml::SVM> poSVM = cv::ml::StatModel::load<cv::ml::SVM>("Model.xml");

    for (int i = 0; i < 2; ++i) {
        // read the positive smaples and calculate the hog feature
        myImageSequence oPositiveReader("D:/Database/01/Negative/", "", "bmp", false);
        oPositiveReader.SetAttribute(myImageSequence::Attribute::PADDING_LENGTH, 6);
        oPositiveReader.SetAttribute(myImageSequence::Attribute::FIRST_NUMBER, 0);

        cv::Mat mPositiveSample;
        std::vector<std::vector<float>> vvfPositiveFeatures;
        while (oPositiveReader >> mPositiveSample) {
            myFeatureExtractor oExtractor(mPositiveSample, BlockSize);
            oExtractor.EnableFeature(myFeatureExtractor::Features::HOG_WITHOUT_NORM);
            oExtractor.EnableFeature(myFeatureExtractor::Features::LBP_8_1_UNIFORM);
            std::vector<std::vector<float>> vvfHOGFeature;

            for (int y = 0; y < mPositiveSample.rows / BlockSize.height; ++y) {
                for (int x = 0; x < mPositiveSample.cols / BlockSize.width; ++x) {
                    std::vector<float> vfFeature;
                    oExtractor.Describe(cv::Point2i(x * BlockSize.width, y * BlockSize.height), vfFeature);
                    vvfHOGFeature.push_back(vfFeature);
                }
            }

            int iIndex = 0;
            cv::Mat mFeature(1, static_cast<int>(vvfHOGFeature.size() * vvfHOGFeature.at(0).size()), CV_32FC1);
            for (const auto& vfHOGFeature : vvfHOGFeature) {
                for (const auto fFeature : vfHOGFeature) {
                    mFeature.at<float>(0, iIndex) = fFeature;
                }
            }
            clock_t start = clock();
            for (int j = 0; j < 10; ++j) {
                
                poSVM->predict(mFeature);
                
            }
            clock_t duration = clock() - start;
            ++iTimes;
            iDuration += duration;
        }
    }

    std::cout << "Total : " << iTimes << std::endl;
    std::cout << "Duration : " << iDuration << std::endl;
    std::cout << "Avg : " << static_cast<double>(iDuration) / static_cast<double>(iTimes) << std::endl;
    return 0;
}