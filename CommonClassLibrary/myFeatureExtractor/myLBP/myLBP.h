#ifndef _MY_LBP_H_
#define _MY_LBP_H_

#include <iostream>
#include <array>
#include <opencv2/core/core.hpp>

#ifdef _DEBUG
#   include <iomanip>
#endif

class myLBP {
public:
    enum class Patterns {
        LBP_8_1 = 0,
        LBP_8_2,
        LBP_16_2,
        LBP_8_1_UNIFORM,
        LBP_8_2_UNIFORM,
        LBP_16_2_UNIFORM
    };

    static const int NUMBER_OF_PATTERNS = 3;    // the number of patterns, not include uniform pattern
    static const int MAX_TRANSITION_TIME = 2;   // the LBP feature wiil be nonuniform if times oftransition (0 -> 1 or 1 -> 0) over it.
    static const int MAX_BIT_LENGTH = 16;       // upper bound of LBP feature length

private:
    cv::Mat m_mImage;
    Patterns m_Pattern;
    cv::Size2i m_BlockSize;
    bool m_bIsUniform;
    int m_iRadius;
    int m_iLength;
    static std::array<std::vector<bool>, myLBP::MAX_BIT_LENGTH / 8> m_avbUniformMap;
    static std::array<std::vector<cv::Point2i>, myLBP::NUMBER_OF_PATTERNS> m_SamplingPoints;

public:
    myLBP(const cv::Mat& mImage, Patterns Pattern, cv::Size2i blockSize = cv::Size2i(8, 8));
    ~myLBP(void);

    void Describe(cv::Point2i Position, std::vector<float>& vfFeature) const;

#ifdef _DEBUG
    void PrintUniformMap(int iLength) const;
#endif

private:
    void Init();
    void SetAttributes(Patterns Pattern);
    unsigned int GetBinNumber(cv::Point2i Position) const;
    static void SetSamplingPoints(void);
    static bool IsUniform(unsigned int iBinNumber, int iLength);
};

#endif