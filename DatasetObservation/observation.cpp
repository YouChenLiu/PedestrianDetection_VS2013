//  applying DFT to sample images and saving beside original image for observing the sample.
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "../CommonClassLibrary/myImageSequence/myImageSequence.h"

void ComputeDFT(cv::Mat& mImg);

int main(void) {
    const std::string sRootPath = "C:/Users/youch/Desktop/Thermal/morning/Negative/";
    const std::string sDstDir = sRootPath + "dft/";
    system((std::string("mkdir \"") + sDstDir + "\"").c_str());
    myImageSequence oReader(sRootPath, "", "bmp", false);
    oReader.SetAttribute(myImageSequence::Attribute::PADDING_LENGTH, 6);

    cv::Mat mImg;
    while (oReader >> mImg) {
        cv::Mat mResult = cv::Mat::zeros(mImg.rows, mImg.cols * 2, mImg.type());
        mImg.copyTo(mResult.colRange(0, mImg.cols));
        ComputeDFT(mImg);
        mImg.copyTo(mResult.colRange(mImg.cols, mResult.cols));
        cv::imwrite(sDstDir + oReader.GetSequenceNumberString() + ".jpg", mResult);
    }

    return 0;
}

void ComputeDFT(cv::Mat& mImg) {
    cv::Mat padded;                            //expand input image to optimal size
    int m = cv::getOptimalDFTSize(mImg.rows);
    int n = cv::getOptimalDFTSize(mImg.cols); // on the border add zero values
    copyMakeBorder(mImg, padded, 0, m - mImg.rows, 0, n - mImg.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    cv::Mat planes[] = { cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F) };
    cv::Mat complexI;
    cv::merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

    cv::dft(complexI, complexI);            // this way the result may fit in the source matrix

    // compute the magnitude and switch to logarithmic scale
    // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
    cv::split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    cv::magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
    cv::Mat magI = planes[0];

    magI += cv::Scalar::all(1);                    // switch to logarithmic scale
    log(magI, magI);

    // crop the spectrum, if it has an odd number of rows or columns
    magI = magI(cv::Rect(0, 0, magI.cols & -2, magI.rows & -2));

    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = magI.cols / 2;
    int cy = magI.rows / 2;

    cv::Mat q0(magI, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    cv::Mat q1(magI, cv::Rect(cx, 0, cx, cy));  // Top-Right
    cv::Mat q2(magI, cv::Rect(0, cy, cx, cy));  // Bottom-Left
    cv::Mat q3(magI, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

    cv::Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);

    normalize(magI, mImg, 0, 255, cv::NORM_MINMAX); // Transform the matrix with float values into a
    // viewable image form (float between values 0 and 1).
}