#include <iostream>
#include "../CommonClassLibrary/myResultVerifier/myResultVerifier.h"

int main(void) {
    myResultVerifier oVerifier("DetectionResult.xml", "GroundTruth.xml");

    oVerifier.CompareXMLResult();

    std::cout << oVerifier.GetDetectionRate() << std::endl;

    system("pause");
    return 0;
}