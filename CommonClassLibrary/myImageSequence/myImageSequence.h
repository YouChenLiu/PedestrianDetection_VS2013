#ifndef _MY_IMAGE_SEQUENCE_H_
#define _MY_IMAGE_SEQUENCE_H_

#include <iostream>
#include <iomanip>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

/*
** ImageSequence is a sequence images reader/writer.
** It can help you access image sequence by setting few parameters.
** You can adjust the parameters by setAttribute method if the default values are not meet your environment.
*/
class myImageSequence {
public:
    enum class Attribute {
        ROOT_PATH = 0,
        PREFIX,
        EXTENSION,
        FIRST_NUMBER,
        PADDING_CHARACTER,
        PADDING_LENGTH,
        OFFSET,
        IS_COLOR
    };

private:
    std::string m_sRootPath;
    std::string m_sPrefix;
    std::string m_sExtension;
    int m_iFirstNumber;
	int m_iOffest;
    int m_iPaddingLength;
    char m_cPaddingCharacter;
    bool m_bIsColor;

public:
    /*
    ** Default constructor
    ** It usually needs to set parameters after creation.
    */
    myImageSequence(void) { init(); }
    
	/*
	** Recommend constructor
	** Setting the root path, prefix, file name extension and isColor.
    ** EX : ImageSequence("C:\\Images\\", "BG-", "jpg", false).
    ** ImageSequence will read/write C:\Images\0000.jpg, 0001.jpg, etc.
    ** NOTE : The isColor parameter only effects when reading.
	*/
    myImageSequence(const std::string& sRootPath, const std::string& sPrefix = "", const std::string& sExtension = "bmp", bool bIsColor = true);
    
    ~myImageSequence(void) {}
    
    /*
    ** Reading a image by ImageSequence
    ** It will return true if success.
    */
	bool readImage(cv::Mat& mImage);
    
    /*
    ** Reading a image by ImageSequence
    ** It will read a image as R value.
    ** It's convenience when declaration a Mat.
    ** Note : It will "not" tell you the reading operation is success or not.
    */
    cv::Mat readImage(void);
    
    /*
    ** Reading a image by ImageSequence
    ** Let you access images like standard input cin.
    ** It will return true if success.
    */
    bool operator>>(cv::Mat& mImage);
    //friend bool operator>>(myImageSequence& lhs, cv::Mat& mImage) { return lhs.operator>>(mImage); }
    
    /*
    ** Write out a image by ImageSequence
    ** It will write image as a image sequence by parameters you set.
    ** It will return true if success.
    */
    bool writeImage(const cv::Mat& mImage);
    
    /*
    ** Write out a image by ImageSequence
    ** Let you save images like standard output cout.
    ** It will return true if success.
    */
    bool operator<<(const cv::Mat& mImage);
    //friend bool operator<<(myImageSequence lhs, const cv::Mat& mImage) { return lhs.operator<<(mImage); }
    
    
    void setAttribute(const Attribute attrbute, const std::string& sValue);
    void setAttribute(const Attribute attrbute, int iValue);
    void setAttribute(const Attribute attrbute, char cValue);
    
    /*
    ** Return a integer which is the current processing number of ImageSequence.
    */
    int getSequenceNumber(void) const { return m_iFirstNumber + m_iOffest; }
    
    /*
    ** Return a string which is the current processing number of ImageSequence.
    ** like "0123" if padding length is 4.
    */
    std::string getSequenceNumberString(void) const;

private:
    void init(void);
    std::string makePath(void);
};

#endif