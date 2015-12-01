#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include "../CommonClassLibrary/myImageSequence/myImageSequence.h"
#include "../CommonClassLibrary/myFeatureExtractor/myFeatureExtractor.h"
#include "../CommonClassLibrary/myResultDumper/myResultDumper.h"

const int iOverlapRatio = 4;
const cv::Size2i NormalizedSize     = { 64, 128 };
const cv::Size2i BlockSize          = { 8, 8 };
const cv::Size2i SmallBoundingBox   = { NormalizedSize.width / 2, NormalizedSize.height / 2 };

int main(void) {
    using namespace std;
    using namespace cv;

    myResultDumper oDumper;
    myImageSequence oImageReader("G:/Share/Database/01/Images/", "", "bmp", false);
    myImageSequence oImageWriter("G:/Share/Database/01/Detection(HOG+LBP)/", "", "bmp");
    Mat mImage;

    SVM oSVM;
    oSVM.load("../Training/Result.xml");
    

    const int iFrameNumberLimit = 100;
    int iFrameNumber = 0;

    while (oImageReader >> mImage) {
        myFeatureExtractor oExtractor(mImage, BlockSize);
        oExtractor.EnableFeature(myFeatureExtractor::Mode::HOG_FEATURE);
        oExtractor.EnableFeature(myFeatureExtractor::Mode::LBP_8_1_UNIFORM);
        for (int y = 0; y < mImage.rows / NormalizedSize.height; ++y) {
            for (int x = 0; x < mImage.cols / NormalizedSize.width; ++x) {
                Size2i BoundingBox = NormalizedSize;
                for (int iShiftY = 0; iShiftY < iOverlapRatio; ++iShiftY) {
                    for (int iShiftX = 0; iShiftX < iOverlapRatio; ++iShiftX) {
                        
                        Point2i BoundingBoxLeftTop(x * BoundingBox.width + iShiftX * (BoundingBox.width / iOverlapRatio), y * BoundingBox.height + iShiftY * (BoundingBox.height / iOverlapRatio));
                        if (BoundingBoxLeftTop.x >= mImage.cols - BoundingBox.width || BoundingBoxLeftTop.y >= mImage.rows - BoundingBox.height) {
                            break;
                        }

                        vector<vector<float>> vvfHOGFeatures;

                        for (int i = 0; i < BoundingBox.height / BlockSize.height; ++i) {
                            for (int j = 0; j < BoundingBox.width / BlockSize.width; ++j) {
                                vector<float> vfFeature;
                                Point2i BlockLeftTop = BoundingBoxLeftTop + Point2i(j * BlockSize.width, i * BlockSize.height);
                                oExtractor.Describe(BlockLeftTop, vfFeature);
                                vvfHOGFeatures.push_back(vfFeature);
                            }
                        }

                        Mat mFeature(1, static_cast<int>(vvfHOGFeatures.size()) * static_cast<int>(vvfHOGFeatures.at(0).size()), CV_32FC1);
                        int col = 0;
                        for (const auto& vfHOGFeature : vvfHOGFeatures) {
                            for (const auto vfFeature : vfHOGFeature) {
                                mFeature.at<float>(col++) = vfFeature;
                            }
                        }

                        int iPredition = static_cast<int>(oSVM.predict(mFeature));
                        if (iPredition == 1) {
                            Rect Box(BoundingBoxLeftTop.x, BoundingBoxLeftTop.y, BoundingBox.width, BoundingBox.height);
                            rectangle(mImage, Box, Scalar(255));
                            oDumper.AddRectangle(BoundingBoxLeftTop.x, BoundingBoxLeftTop.y, BoundingBox.width, BoundingBox.height);
                        }
                    }
                }
                
            }
        }
        oDumper.GoNextFrame();
        //imshow("Detection", mImage);
        //waitKey(0);
        oImageWriter << mImage;
        if (iFrameNumber++ >= iFrameNumberLimit) {
            break;
        }
    }

    oDumper.Save("DetectionResult.xml");
    
    return 0;
}