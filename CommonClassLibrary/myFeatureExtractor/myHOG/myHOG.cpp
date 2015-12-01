#include "myHOG.h"

const float myHOG::m_fUnimportantValue = 1e-6f;

myHOG::myHOG(const cv::Mat& mImage, cv::Size2i blockSize, int iInterval) {
    Init();
    m_mHorizontalGradientImage = mImage * DIPKernel(DIPKernel::Types::SIMPLE_X);
    m_mVerticalGradientImage = mImage * DIPKernel(DIPKernel::Types::SIMPLE_Y);
    m_BlockSize = blockSize;
    m_iInterval = iInterval;
}

myHOG::~myHOG(void) {}

void myHOG::Init(void) {
    m_mHorizontalGradientImage = m_mVerticalGradientImage = cv::Mat();
    m_iInterval = 0;
    m_BlockSize = cv::Size2i(0, 0);
    m_NormalizedType = NormalizedTypes::NONE;
}

void myHOG::Normalize(std::vector<float>& vfFeature, NormalizedTypes type) {
    float fNormFactor = 0.0f;
    switch (type) {
    case NormalizedTypes::NONE:
        break;
    case NormalizedTypes::L1_NORM:
        break;
    case NormalizedTypes::L2_NORM:
        for (auto fFeature : vfFeature) {
            fNormFactor += fFeature * fFeature;
        }
        fNormFactor = sqrt(fNormFactor + m_fUnimportantValue * m_fUnimportantValue);

        for (auto& fFeature : vfFeature) {
            fFeature /= fNormFactor;
        }
        break;
    default:
        std::cout << "not support normalization method" << std::endl;
        break;
    }
}

void myHOG::Describe(cv::Point2i Position, std::vector<float>& vfHogFeature) const {
    cv::Size2i CellSize = cv::Size2i(m_BlockSize.width / 2, m_BlockSize.height / 2);
    
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            cv::Point2i RoiLocation = Position + cv::Point2i(j * CellSize.width, i * CellSize.height);
            DescribeCell(RoiLocation, vfHogFeature);
        }
    }

    if (m_NormalizedType != NormalizedTypes::NONE) {
        myHOG::Normalize(vfHogFeature, m_NormalizedType);
    }
}

void myHOG::DescribeCell(const cv::Point2i Position, std::vector<float>& vfHogFeature) const {
    std::vector<float> vfBins(180 / m_iInterval, 0.0);
    cv::Size2i CellSize = cv::Size2i(m_BlockSize.width / 2, m_BlockSize.height / 2);
    cv::Mat mHorizontalRoi = cv::Mat(m_mHorizontalGradientImage, cv::Rect(Position, CellSize));
    cv::Mat mVerticalRoi = cv::Mat(m_mVerticalGradientImage, cv::Rect(Position, CellSize));
    
    for (int y = 0; y < CellSize.height; ++y) {
        for (int x = 0; x < CellSize.width; ++x) {
            float fOrientation = atan2f(mVerticalRoi.at<short>(y, x), mHorizontalRoi.at<short>(y, x)) * 180.0f / static_cast<float>(CV_PI);
            float fMagnitude = sqrtf(static_cast<float>(mVerticalRoi.at<short>(y, x) * mVerticalRoi.at<short>(y, x) + 
                                                        mHorizontalRoi.at<short>(y, x) * mHorizontalRoi.at<short>(y, x)));
            if (fOrientation < 0) {
                fOrientation += 180;
            } else if (fOrientation > 180) {
                fOrientation -= 180;
            }
            
            int target = static_cast<int>(fOrientation / m_iInterval);
            if (target >= static_cast<int>(vfBins.size())) {
                target = static_cast<int>(vfBins.size()) - 1;
            }
            vfBins[target] += fMagnitude;
        }
    }

    for (auto bin : vfBins) {
        vfHogFeature.push_back(bin);
    }
}