#ifndef _MY_FEATURE_EXTRACTOR_H_
#define _MY_FEATURE_EXTRACTOR_H_

#include <iostream>
#include <iomanip>
#include <memory>
#include "myFeatureBase.h"
#include "myHOG/myHOG.h"
#include "myLBP/myLBP.h"

#ifndef NDEBUG
#   include <fstream>
#endif

class myFeatureExtractor {
public:

private:
    cv::Size2i m_BlockSize;
    cv::Mat m_mImage;
    std::unique_ptr<myHOG> m_poHOG;
    std::unique_ptr<myLBP> m_poLBP;

public:
    myFeatureExtractor(cv::Mat& mImage, cv::Size2i BlockSize = cv::Size2i(20, 20));
    myFeatureExtractor(IplImage* pImage, CvSize BlockSize = cvSize(20, 20));
    ~myFeatureExtractor(void);

    void Describe(cv::Point2i Position, std::vector<float>& vfFeature) const;
    void Describe(int x, int y, std::vector<float>& vfFeature) const { Describe(cv::Point2i(x, y), vfFeature); }
    
    void EnableFeature(int iFeature);
    void SetBlockSize(cv::Size2i Size) { m_BlockSize = Size; }
    void SetBlockSize(int Width, int Height) { SetBlockSize(cv::Size2i(Width, Height)); }

private:
    void Init(void);
    
};

#endif