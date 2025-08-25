

#include "ImageStitcher.hpp"
#include <direct.h>


int main() {
    _chdir(R"(C:\Users\Lorenzo\Desktop\TESI\test_folder\p2)");
     
    ImageStitcher stitcher;

    stitcher.newProcess();
    stitcher.addImage("a.dcm");
    stitcher.addImage("b.dcm");
    stitcher.addImage("c.dcm");
    stitcher.addImage("d.dcm");

    stitcher.stitch(
        "abcd",
        ImageStitcher::DetectorAlgorithm::SIFT,
        ImageStitcher::MatchingAlgorithm::FLANN,
        ImageStitcher::HomographyMethod::RANSAC);

    return 0;
}