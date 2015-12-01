#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "../CommonClassLibrary/myKitDef.h"
#include "../CommonClassLibrary/myIPDef.h"
#include "../CommonClassLibrary/myImageSequence/myImageSequence.h"
#include "../CommonClassLibrary/myFeatureExtractor/myFeatureExtractor.h"
#include <vector>
#include <chrono>
int main(void) {
    cv::Mat mImg = cv::imread("BG.bmp", cv::IMREAD_GRAYSCALE);
    myFeatureExtractor extractor(mImg, cv::Size2i(20, 20));
    extractor.EnableFeature(myFeatureExtractor::Mode::HOG_FEATURE);
    std::vector<float> v;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i) {
        extractor.Describe(cv::Point2i(0, 0), v);
    }
    auto duration = std::chrono::high_resolution_clock::now() - start;
    auto microsecond = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    std::cout << microsecond / 1000 << std::endl;
    /*
    cv::Mat mBGImage = cv::imread("BG.bmp", cv::IMREAD_COLOR);
    myBGModel oBGModel(640, 480);
    oBGModel.DoMakeImpl(myBGModel::ID_ABSPIXELDIFFERENCE);
    oBGModel.Model(mBGImage);

    myImageSequence oInputSeq("C:/Database/test/", "", "bmp");
    oInputSeq.setAttribute(myImageSequence::Attribute::PADDING_LENGTH, 5);

    cv::VideoWriter oWriter("ABS.avi", CV_FOURCC('F', 'M', 'P', '4'), 60, mBGImage.size(), false);

    cv::Mat mImg;
    cv::Mat mFGMask(mImg.size(), CV_8UC1);
    
    while (oInputSeq >> mImg) {
        oBGModel.Subtract(mImg, mFGMask);
        oBGModel.Update(mImg);
        oWriter << mFGMask;
        std::cout << oInputSeq.getSequenceNumber() << std::endl;
    }
    */
    /*
    cv::VideoCapture oGMMVideo("GMM.avi");
    cv::VideoCapture oABSVideo("ABS.avi");
    myImageSequence oInputSeq("C:/Database/test/", "", "bmp");
    oInputSeq.setAttribute(myImageSequence::Attribute::PADDING_LENGTH, 5);

    cv::Mat mPreSourceImg = cv::imread("BG.bmp", cv::IMREAD_COLOR);

    cv::VideoWriter oOutputVideo("Combine.avi", CV_FOURCC('F', 'M', 'P', '4'), 60, cv::Size2i(1280, 960));
    
    while (true) {
        cv::Mat mGMMImg;
        cv::Mat mABSImg;
        cv::Mat mSourceImg;
        cv::Mat mDiffImg;

        if (oGMMVideo.read(mGMMImg) && oABSVideo.read(mABSImg) && oInputSeq.readImage(mSourceImg)) {
            cv::Mat mImg = cv::Mat::zeros(mSourceImg.size() * 2, mSourceImg.type());
            mSourceImg.copyTo(mImg.rowRange(0, 480).colRange(0, 640));
            mGMMImg.copyTo(mImg.rowRange(0, 480).colRange(640, 1280));
            mABSImg.copyTo(mImg.rowRange(480, 960).colRange(0, 640));
            cv::Canny(mABSImg, mDiffImg, 50, 100);
            cv::cvtColor(mDiffImg, mDiffImg, CV_GRAY2BGR);
            mDiffImg.copyTo(mImg.rowRange(480, 960).colRange(640, 1280));
            //mPreSourceImg = mSourceImg;
            oOutputVideo << mImg;
        } else {
            break;
        }

    }
    */
    return 0;
}