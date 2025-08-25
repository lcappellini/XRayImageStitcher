#include <string>
#include <vector>
#include <opencv2/opencv.hpp>


class ImageStitcher {

public:
    enum class HomographyMethod {
        LSQ = 0,
        RANSAC = cv::RANSAC,
        LMEDS = cv::LMEDS,
        PROSAC = cv::RHO
    };

    enum class DetectorAlgorithm {
        SIFT,
        SURF,
        ORB,
        BRISK,
        AKAZE,
        KAZE,
        SUPERPOINT
    };

    enum class MatchingAlgorithm {
        BRUTEFORCE,
        FLANN,
        SUPERGLUE
    };

    enum class DescriptorType {
        FLOAT,
        BINARY
    };

    struct CroppingRect {
        int leftCrop = 0;
        int topCrop = 0;
        int rightCrop = 0;
        int bottomCrop = 0;

        CroppingRect() = default;
        CroppingRect(int leftCrop, int topCrop, int rightCrop, int bottomCrop)
            : leftCrop(leftCrop), topCrop(topCrop), rightCrop(rightCrop), bottomCrop(bottomCrop) {
        }
        bool isEmpty() const {
            return leftCrop == 0 && topCrop == 0 && rightCrop == 0 && bottomCrop == 0;
        }
    };

private:
    struct ImageData {
        //dcm file
        std::string imageFilename;

        //raw data
        void* rawImageData;
        int width, height, channels, depth;

        //openCV matrix
        cv::Mat matrix;

        enum class Type { FILENAME, RAW, CV_MAT };
        Type type;

        ImageData * raw = nullptr;

        CroppingRect croppingRect = CroppingRect();
    };

private:
    std::string superpoint_model_path = R"(C:\Users\Lorenzo\Desktop\TESI\Tesi Test Project\Tesi Test Project\Tesi Test Project\superglue\superpoint.onnx)";
    std::string superglue_model_path = R"(C:\Users\Lorenzo\Desktop\TESI\Tesi Test Project\Tesi Test Project\Tesi Test Project\superglue\superglue_indoor.onnx)";

    std::vector<ImageData> sources;

    float portionToAnalyse = 1.0f / 4.0f; //TODO should i make it customizable for every pair of images (?)

    cv::Mat load_image(const ImageData& imgData);
    bool get_img_from_source(ImageData& imgData, cv::Mat& out_img_work, cv::Mat& out_img_res, cv::Mat& out_img_raw);
    cv::Mat dicom_to_cv_8(std::string path);
    cv::Mat cv16_to_cv8(cv::Mat img16);
    cv::Mat dicom_to_cv_16(std::string path);

    cv::Mat crop_image(const cv::Mat& img, const CroppingRect& crop);

    bool apply_alpha_blending(const cv::Mat& img1_res, const cv::Mat& img2_res, cv::Mat& gray_out, std::vector<cv::Point2f>& corners_transformed, int overlap, cv::Mat& H);

public:
	void newProcess();
	bool addImage(std::string dcmFilename, CroppingRect croppingRect = CroppingRect());
	bool addImage(void* rawPixels, int w, int h, int ch, int depth, CroppingRect croppingRect = CroppingRect());
	bool addImage(const cv::Mat& mat, CroppingRect croppingRect = CroppingRect());

    bool addImageWithRaw(std::string dcmFilename, std::string dcmFilenamer = "", CroppingRect croppingRect = CroppingRect());
    //bool addImageWithRaw(void* rawPixels, int w, int h, int ch, int depth, void* rawPixelsr, int wr, int hr, int chr, int depthr);
    //bool addImageWithRaw(const cv::Mat& mat, const cv::Mat& matr);

    unsigned int getNumberOfImages() const;

    bool stitch(std::string outFilename,
        DetectorAlgorithm detectorAlgorithm = DetectorAlgorithm::SIFT,
        MatchingAlgorithm matchingAlgorithm = MatchingAlgorithm::BRUTEFORCE,
        HomographyMethod homographyMethod = HomographyMethod::RANSAC);
};