
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#define DESC_DIM 256

//singleton design patter: no reed to reload the model from the file every time
class SuperPoint : public cv::Feature2D {
private:

    static cv::Ptr<cv::Feature2D> instance; //ptr to the singleton instance

    Ort::Env env;
    Ort::Session* session;    

    float kp_threshold;

    SuperPoint(std::string model_file_path, float keypoint_threshold) {
        kp_threshold = keypoint_threshold;
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "SuperPoint");

        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        std::wstring ws(model_file_path.begin(), model_file_path.end());
        const wchar_t* model_path = ws.c_str();
        session = new Ort::Session(env, model_path, session_options);
    };

    std::vector<Ort::Value> runSession(cv::Mat img);

    void parse_superpoint_output(std::vector<Ort::Value>& superpoint_tensors,
        std::vector<float>& keypoints,
        std::vector<float>& scores,
        std::vector<float>& descriptors);

    void superpoint_raw_to_cv(
        const std::vector<float>& keypoints,
        const std::vector<float>& scores,
        const std::vector<float>& descriptors, 
        std::vector<cv::KeyPoint>& cv_kpts,
        cv::Mat& cv_descs);

public:
    SuperPoint(const SuperPoint&) = delete;
    SuperPoint& operator=(const SuperPoint&) = delete;

    static cv::Ptr<cv::Feature2D> create(std::string model_file_path, float keypoint_threshold) {
        if (!instance) {
            instance = cv::Ptr<SuperPoint>(new SuperPoint(model_file_path, keypoint_threshold));
        }
        return instance;
    };

    Ort::Session& getSession() {
        return *session;
    }

    void detect(cv::InputArray image,
        std::vector<cv::KeyPoint>& keypoints,
        cv::InputArray mask = cv::noArray()) override;

    void compute(cv::InputArray image,
        std::vector<cv::KeyPoint>& keypoints,
        cv::OutputArray descriptors) override;

    void detectAndCompute(cv::InputArray image,
        cv::InputArray mask,
        std::vector<cv::KeyPoint>& keypoints,
        cv::OutputArray descriptors,
        bool useProvidedKeypoints = false) override;
};


