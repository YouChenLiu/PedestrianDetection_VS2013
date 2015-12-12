#include <fstream>
#include <iomanip>
#include <opencv2/highgui.hpp>
#include "../CommonClassLibrary/myFeatureExtractor/myFeatureExtractor.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        return EXIT_FAILURE;
    }
    
    std::string sImgPath = argv[1];
    std::string sFileName = sImgPath.substr(sImgPath.rfind("\\") + 1, sImgPath.rfind(".jpg") - sImgPath.rfind("\\") - 1);
    std::string sFilePath = sImgPath.substr(0, sImgPath.rfind("\\") + 1) + sFileName + ".txt";
    std::ofstream oFile(sFilePath);
    cv::Mat mImg = cv::imread(sImgPath, cv::IMREAD_GRAYSCALE);
    
    cv::Size2i BlockSize(8, 8);
    myFeatureExtractor oExtractor(mImg, BlockSize);
    oExtractor.EnableFeature(myFeatureExtractor::Features::HOG_WITHOUT_NORM);
    oExtractor.EnableFeature(myFeatureExtractor::Features::LBP_8_1_UNIFORM);
    int iFeatureIndex = 0;
    for (int y = 0; y < mImg.rows; y += BlockSize.height) {
        for (int x = 0; x < mImg.cols; x += BlockSize.width) {
            std::vector<float> vfFeature;
            oExtractor.Describe(cv::Point2i(x, y), vfFeature);
            
            oFile.precision(6);
            oFile << std::setfill('0') << std::setw(5) << std::right << iFeatureIndex++ << ": ";
            oFile << std::setfill(' ');
            int i = 0;
            for (auto feature : vfFeature) {
                std::stringstream oText;
                oText << i++ << ":" << feature;
                oFile << std::setw(12) << std::left << oText.str();
            }
            oFile << std::endl;
        }
    }

    return 0;
}