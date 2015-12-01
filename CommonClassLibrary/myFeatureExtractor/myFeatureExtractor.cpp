#include "myFeatureExtractor.h"

myFeatureExtractor::myFeatureExtractor(IplImage* pImage, CvSize BlockSize) {
    Init();
    m_mImage = cv::cvarrToMat(pImage, true);
    m_BlockSize = cv::Size2i(BlockSize.width, BlockSize.height);
}

myFeatureExtractor::myFeatureExtractor(cv::Mat& mImage, cv::Size2i BlockSize) {
    Init();
    m_mImage = mImage;
    m_BlockSize = BlockSize;
}

myFeatureExtractor::~myFeatureExtractor(void) {}

void myFeatureExtractor::Init(void) {
    m_BlockSize = cv::Size2i(0, 0);
    m_mImage = cv::Mat();
    m_poHOG = nullptr;
    m_poLBP = nullptr;
}

void myFeatureExtractor::Describe(cv::Point2i Position, std::vector<float>& vfFeature) const {
    using namespace std;
    vfFeature.clear();
    
    vector<float> vfHOGFeature;
    if (m_poHOG != nullptr) {
        m_poHOG->Describe(Position, vfHOGFeature);
    }
    for (auto feature : vfHOGFeature) {
        vfFeature.push_back(feature);
    }
    
    vector<float> vfLBPFeature;
    if (m_poLBP != nullptr) {
        m_poLBP->Describe(Position, vfLBPFeature);
    }
    for (auto feature : vfLBPFeature) {
        vfFeature.push_back(feature);
    }

#   ifndef NDEBUG
    static int iFeatureIndex = 0;
    static ofstream oFeatureFile("Feature.txt");
    if (oFeatureFile.is_open() == true) {
        oFeatureFile.precision(6);
        oFeatureFile << setfill('0') << setw(5) << std::right << iFeatureIndex++ << ": ";
        oFeatureFile << setfill(' ');
        int i = 0;
        for (auto feature : vfFeature) {
            stringstream oText;
            oText << i++ << ":" << feature;
            oFeatureFile << setw(12) <<  std::left << oText.str();
        }
        oFeatureFile << endl;
    }
#   endif
}

void myFeatureExtractor::EnableFeature(Mode mode) {
    using namespace std;
    switch (mode) {
    case Mode::HOG_FEATURE:
        m_poHOG = make_unique<myHOG>(myHOG(m_mImage, m_BlockSize));
        break;
    case Mode::LBP_8_1:
        m_poLBP = make_unique<myLBP>(myLBP(m_mImage, myLBP::Patterns::LBP_8_1, m_BlockSize));
        break;
    case Mode::LBP_FEATURE:
    case Mode::LBP_8_1_UNIFORM:
        m_poLBP = make_unique<myLBP>(myLBP(m_mImage, myLBP::Patterns::LBP_8_1_UNIFORM, m_BlockSize));
        break;
    case Mode::LBP_8_2:
        m_poLBP = make_unique<myLBP>(myLBP(m_mImage, myLBP::Patterns::LBP_8_2, m_BlockSize));
        break;
    case Mode::LBP_8_2_UNIFORM:
        m_poLBP = make_unique<myLBP>(myLBP(m_mImage, myLBP::Patterns::LBP_8_2_UNIFORM, m_BlockSize));
        break;
    case Mode::LBP_16_2:
        m_poLBP = make_unique<myLBP>(myLBP(m_mImage, myLBP::Patterns::LBP_16_2, m_BlockSize));
        break;
    case Mode::LBP_16_2_UNIFORM:
        m_poLBP = make_unique<myLBP>(myLBP(m_mImage, myLBP::Patterns::LBP_16_2_UNIFORM, m_BlockSize));
        break;
    default:
        cout << "Not valid feature" << endl;
        break;
    }
}