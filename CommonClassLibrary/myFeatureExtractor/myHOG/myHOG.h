#ifndef _MY_HOG_H_
#define _MY_HOG_H_

#include <iostream>
#include <array>
#include <vector>
#include <opencv2/core/core.hpp>
#include "../DIPKernel/DIPKernel.h"

class myHOG {
public:
    enum class NormalizedTypes {
        NONE = -1,
        L1_NORM = 0,
        L1_SQRT,
        L2_NORM,
        L2_SQRT
    };
private:
    cv::Mat m_mHorizontalGradientImage;
    cv::Mat m_mVerticalGradientImage;
    int m_iInterval;
    cv::Size2i m_BlockSize;
    static const float m_fUnimportantValue;
    NormalizedTypes m_NormalizedType;

public:
    myHOG(const cv::Mat& mImage, cv::Size2i blockSize = cv::Size2i(8, 8), int iInterval = 20);
    ~myHOG(void);

    void Describe(cv::Point2i Position, std::vector<float>& vfHogFeature) const;
    void SetNormalizedType(NormalizedTypes type) { m_NormalizedType = type; }

private:
    void Init(void);
    void DescribeCell(const cv::Point2i Position, std::vector<float>& vfHogFeature) const;
    void static Normalize(std::vector<float>& vfFeature, NormalizedTypes type);
};

#endif