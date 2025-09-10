
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>


class SuperGlue : public cv::DescriptorMatcher {
private:
    static cv::Ptr<SuperGlue> instance;

    Ort::Env env;
    Ort::Session* session;

    SuperGlue(std::string model_file_path, float match_threshold) {
        m_threshold = match_threshold;

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "SuperGlue");

        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(4);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        std::wstring ws(model_file_path.begin(), model_file_path.end());
        const wchar_t* model_path = ws.c_str();
        session = new Ort::Session(env, model_path, session_options);
    };

    cv::Mat lastDescriptors;

    std::vector<Ort::Value> runSession(cv::Mat img);

    std::vector<float> keypoints0;
    std::vector<float> keypoints1;
    std::vector<float> scores0;
    std::vector<float> scores1;
    int img0W = 0;
    int img0H = 0;
    int img1W = 0;
    int img1H = 0;
    int desc_dim = 0;
    
    float m_threshold = 0;

    void parse_superglue_output(std::vector<Ort::Value>& superglue_tensors, 
        std::vector<cv::DMatch>& matches);

public:
    SuperGlue(const SuperGlue&) = delete;
    SuperGlue& operator=(const SuperGlue&) = delete;


    static cv::Ptr<SuperGlue> create(const std::string& model_file_path, float match_threshold=0.02f) {
        if (instance.empty()) {
            instance = cv::Ptr<SuperGlue>(new SuperGlue(model_file_path, match_threshold));
        }
        instance->clear();
        return instance;
    }

    Ort::Session& getSession() {
        return *session;
    }

    void init( //TODO RENAME THIS
        const std::vector<cv::KeyPoint>& kpts0,
        const std::vector<cv::KeyPoint>& kpts1,
        int img0_width, int img0_height, int img1_width, int img1_height,
        int offsetx = 0, int offsety = 0,
        int desc_dim = 256
    );

    // methods from base class
    virtual void knnMatchImpl(
        cv::InputArray queryDescriptors,
        std::vector<std::vector<cv::DMatch>>& matches,
        int k,
        cv::InputArrayOfArrays masks = cv::noArray(),
        bool compactResult = false);

    virtual void radiusMatchImpl(
        cv::InputArray queryDescriptors,
        std::vector<std::vector<cv::DMatch>>& matches,
        float maxDistance,
        cv::InputArrayOfArrays masks = cv::noArray(),
        bool compactResult = false);

    virtual cv::Ptr<DescriptorMatcher> clone(bool emptyTrainData = false) const;

    virtual bool isMaskSupported() const;
    //


};