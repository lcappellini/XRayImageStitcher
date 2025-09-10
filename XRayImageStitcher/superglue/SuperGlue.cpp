
#include "SuperGlue.hpp"

cv::Ptr<SuperGlue> SuperGlue::instance = nullptr;


void SuperGlue::parse_superglue_output(std::vector<Ort::Value>& superglue_tensors, std::vector<cv::DMatch>& matches) {
    float* scores = superglue_tensors[0].GetTensorMutableData<float>();
    auto output_shape = superglue_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    int N = output_shape[1];
    int M = output_shape[2];

    std::vector<int> indices0(N, -1);
    std::vector<int> indices1(M, -1);
    std::vector<float> max0(N, -std::numeric_limits<float>::infinity());
    std::vector<float> max1(M, -std::numeric_limits<float>::infinity());

    for (int i = 0; i < N - 1; ++i) {
        for (int j = 0; j < M - 1; ++j) {
            float val = scores[i * M + j];
            if (val > max0[i]) {
                max0[i] = val;
                indices0[i] = j;
            }

            if (val > max1[j]) {
                max1[j] = val;
                indices1[j] = i;
            }
        }
    }

    // mutual check
    for (int i = 0; i < N; i++) {
        int j = indices0[i];
        if (j >= 0 && indices1[j] == i) {
            float s = std::exp(max0[i]);
            if (s > m_threshold) {
                matches.push_back({ i, j, s });
                //mscores0[i] = s;
            }
            else {
                indices0[i] = -1;
            }
        }
        else {
            indices0[i] = -1;
        }
    }
}


std::vector<float> normalize_keypoints(const std::vector<float>& kpts, int width, int height) {
    std::vector<float> norm_kpts;
    norm_kpts.reserve(kpts.size());

    float cx = width / 2.0f;
    float cy = height / 2.0f;
    float scale = 0.7f * std::max(width, height);

    for (size_t i = 0; i < kpts.size(); i += 2) {
        float x = (kpts[i] - cx) / scale;
        float y = (kpts[i + 1] - cy) / scale;
        norm_kpts.push_back(x);
        norm_kpts.push_back(y);
    }
    return norm_kpts;
}


std::vector<Ort::Value> SuperGlue::runSession(cv::Mat img) {
    std::vector<Ort::Value> vec;
    return vec;
}


void SuperGlue::knnMatchImpl(cv::InputArray queryDescriptors, std::vector<std::vector<cv::DMatch>>& matches, int k, cv::InputArrayOfArrays masks, bool compactResult) {
    // prepare input data
    std::vector<float> descriptors0, descriptors1;
    int64_t N0 = (int64_t)scores0.size();
    int64_t N1 = (int64_t)scores1.size();

    cv::Mat query = queryDescriptors.getMat();
    int desc0_dim = query.cols;
    cv::Mat train = getTrainDescriptors()[0];
    int desc1_dim = train.cols;

    if (query.type() != CV_32F) {
        std::cout << "SperGlue is not compatible with non-float32 descriptors" << std::endl;
        throw std::runtime_error("SperGlue is not compatible with non-float32 descriptors");
    }

    if (desc0_dim != 256 || desc1_dim != 256) { //TODO is there no workarounds? zero-padding?
        std::cout << "SuperGlue requires descriptors of size 256 (float32)" << std::endl;
        throw std::runtime_error("SuperGlue requires descriptors of size 256 (float32)");
    }

    //std::cout << query.size() << std::endl;
    //std::cout << train.size() << std::endl;

    descriptors0.reserve(N0 * 256);
    for (int c = 0; c < desc0_dim; c++) {
        for (int r = 0; r < N0; r++) {
            descriptors0.push_back(query.at<float>(r, c));
        }
        //for (int k = desc0_dim; k < 256; k++) {
        //    descriptors0.push_back(0.0f);
        //}
    }


    descriptors1.reserve(N1 * 256);
    for (int c = 0; c < desc1_dim; c++) {
        for (int r = 0; r < N1; r++) {
            descriptors1.push_back(train.at<float>(r, c));
        }
        //for (int k = desc1_dim; k < 256; k++) {
        //    descriptors1.push_back(0.0f);
        //}
    }
    std::vector<float> keypoints0_norm, keypoints1_norm;
    keypoints0_norm = normalize_keypoints(keypoints0, img0W, img0H);
    keypoints1_norm = normalize_keypoints(keypoints1, img1W, img1H);
    //

    // set input data
    std::vector<Ort::Value> input_tensors;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    int64_t dims_kpts_0[3] = { 1, N0, 2 };
    int64_t dims_scores_0[2] = { 1, N0 };
    int64_t dims_desc_0[3] = { 1, (int64_t)desc_dim, N0 };
    input_tensors.push_back(Ort::Value::CreateTensor<float>(memory_info, keypoints0_norm.data(), keypoints0_norm.size(), dims_kpts_0, 3));
    input_tensors.push_back(Ort::Value::CreateTensor<float>(memory_info, scores0.data(), N0, dims_scores_0, 2));
    input_tensors.push_back(Ort::Value::CreateTensor<float>(memory_info, descriptors0.data(), descriptors0.size(), dims_desc_0, 3));
    int64_t dims_kpts_1[3] = { 1, N1, 2 };
    int64_t dims_scores_1[2] = { 1, N1 };
    int64_t dims_desc_1[3] = { 1, (int64_t)desc_dim, N1 };
    input_tensors.push_back(Ort::Value::CreateTensor<float>(memory_info, keypoints1_norm.data(), keypoints1_norm.size(), dims_kpts_1, 3));
    input_tensors.push_back(Ort::Value::CreateTensor<float>(memory_info, scores1.data(), N1, dims_scores_1, 2));
    input_tensors.push_back(Ort::Value::CreateTensor<float>(memory_info, descriptors1.data(), descriptors1.size(), dims_desc_1, 3));
    //

    // run inference
    const char* input_names[] = { "keypoints_0", "scores_0", "descriptors_0", "keypoints_1", "scores_1", "descriptors_1" };
    const char* output_names[] = { "scores" };
    std::vector<Ort::Value> output_tensors = session->Run(Ort::RunOptions{ nullptr }, input_names, input_tensors.data(), 6, output_names, 1);
    //

    // get the output
    std::vector<cv::DMatch> matches_vec;
    matches.clear();
    parse_superglue_output(output_tensors, matches_vec);
    for (const auto& m : matches_vec) {
        if (m.queryIdx >= N0)
            continue;
        if (m.trainIdx >= N1)
            continue;
        matches.push_back(std::vector<cv::DMatch>{ m });
    }
    //
}

void SuperGlue::radiusMatchImpl(cv::InputArray queryDescriptors, std::vector<std::vector<cv::DMatch>>& matches, float maxDistance, cv::InputArrayOfArrays masks, bool compactResult) {
    throw std::runtime_error("Not implemented");
}

bool SuperGlue::isMaskSupported() const {
    return false;
}

void SuperGlue::init(const std::vector<cv::KeyPoint>& kpts0, const std::vector<cv::KeyPoint>& kpts1, int img0_width, int img0_height, int img1_width, int img1_height, int offsetx, int offsety, int descriptor_dim) {
    keypoints0.clear();
    keypoints1.clear();
    scores0.clear();
    scores1.clear();

    scores0.reserve(kpts0.size() * 2);
    scores1.reserve(kpts1.size() * 2);
    scores0.reserve(kpts0.size());
    scores1.reserve(kpts1.size());

    for (const auto& kp : kpts0) {
        keypoints0.push_back(kp.pt.x - offsetx);
        keypoints0.push_back(kp.pt.y - offsety);
        scores0.push_back(kp.response);
    }

    for (const auto& kp : kpts1) {
        keypoints1.push_back(kp.pt.x);
        keypoints1.push_back(kp.pt.y);
        scores1.push_back(kp.response);
    }

    img0W = img0_width;
    img0H = img0_height;
    img1W = img1_width;
    img1H = img1_height;
    desc_dim = descriptor_dim;
}

cv::Ptr<cv::DescriptorMatcher> SuperGlue::clone(bool emptyTrainData) const {
    return instance;
};