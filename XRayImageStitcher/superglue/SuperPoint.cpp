
#include "SuperPoint.hpp"

cv::Ptr<cv::Feature2D> SuperPoint::instance = nullptr;


void sample_descriptors(
    const std::vector<float>& keypoints,
    const float* descriptors_feat,
    int C, int H_feat, int W_feat,
    int s,
    std::vector<float>& descriptors_out)
{
    size_t num_kp = keypoints.size() / 2;
    descriptors_out.resize(num_kp * C);

    for (size_t k = 0; k < num_kp; ++k) {
        float x = keypoints[2 * k];
        float y = keypoints[2 * k + 1];

        float xf = (x - s / 2.0f + 0.5f) / (W_feat * s - s / 2.0f - 0.5f) * (W_feat - 1);
        float yf = (y - s / 2.0f + 0.5f) / (H_feat * s - s / 2.0f - 0.5f) * (H_feat - 1);

        int x0 = std::floor(xf);
        int x1 = std::min(x0 + 1, W_feat - 1);
        int y0 = std::floor(yf);
        int y1 = std::min(y0 + 1, H_feat - 1);

        float dx = xf - x0;
        float dy = yf - y0;

        float norm = 0.0f;
        for (int c = 0; c < C; ++c) {
            float v00 = descriptors_feat[c * H_feat * W_feat + y0 * W_feat + x0];
            float v01 = descriptors_feat[c * H_feat * W_feat + y1 * W_feat + x0];
            float v10 = descriptors_feat[c * H_feat * W_feat + y0 * W_feat + x1];
            float v11 = descriptors_feat[c * H_feat * W_feat + y1 * W_feat + x1];

            float v0 = v00 * (1 - dy) + v01 * dy;
            float v1 = v10 * (1 - dy) + v11 * dy;
            float val = v0 * (1 - dx) + v1 * dx;

            descriptors_out[k * C + c] = val;
            norm += val * val;
        }

        norm = std::sqrt(norm) + 1e-10f;
        for (int c = 0; c < C; ++c)
            descriptors_out[k * C + c] /= norm;
    }
}


void SuperPoint::parse_superpoint_output(std::vector<Ort::Value>& superpoint_tensors,
    std::vector<float>& keypoints,
    std::vector<float>& scores,
    std::vector<float>& descriptors) {
    float* scores_data = superpoint_tensors[0].GetTensorMutableData<float>();
    auto scores_shape = superpoint_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    int H = scores_shape[1];
    int W = scores_shape[2];

    float* desc_data = superpoint_tensors[1].GetTensorMutableData<float>();
    auto desc_shape = superpoint_tensors[1].GetTensorTypeAndShapeInfo().GetShape();
    int desc_dim = desc_shape[1];
    int H_feat = desc_shape[2];
    int W_feat = desc_shape[3];

    std::vector<int> keypoint_indices;

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float sc = scores_data[y * W + x];
            if (sc > kp_threshold) {
                keypoints.push_back((float)x);
                keypoints.push_back((float)y);
                scores.push_back(sc);
                keypoint_indices.push_back(y * W + x);
            }
        }
    }
    int N = scores.size();
    sample_descriptors(keypoints, desc_data, DESC_DIM, H_feat, W_feat, 8, descriptors);
}


void SuperPoint::detect(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::InputArray mask)
{
    throw std::runtime_error("Not implemented");
}


void SuperPoint::compute(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors)
{
    throw std::runtime_error("Not implemented");
}


void SuperPoint::superpoint_raw_to_cv(
    const std::vector<float>& keypoints,
    const std::vector<float>& scores,
    const std::vector<float>& descriptors,
    std::vector<cv::KeyPoint>& cv_kpts,
    cv::Mat& cv_descs)
{
    size_t num_kp = keypoints.size() / 2;
    cv_kpts.clear();
    cv_descs = cv::Mat(static_cast<int>(num_kp), DESC_DIM, CV_32F);

    for (size_t i = 0; i < num_kp; ++i) {
        float x = keypoints[2 * i];
        float y = keypoints[2 * i + 1];
        float score = scores[i];

        cv::KeyPoint kp;
        kp.pt = cv::Point2f(x, y);
        kp.size = 1.f;
        kp.response = score;
        kp.angle = -1;
        kp.octave = 0;
        cv_kpts.push_back(kp);

        for (int64_t j = 0; j < DESC_DIM; j++) {
            cv_descs.at<float>(static_cast<int>(i), j) = descriptors[i * DESC_DIM + j];
        }
    }
}

void SuperPoint::detectAndCompute(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors, bool useProvidedKeypoints)
{
    keypoints.clear();
    cv::Mat img = image.getMat();
    int h = img.rows;
    int w = img.cols;

    // apply mask (crop image)
    int first_row = -1, last_row = h;
    int first_col = -1, last_col = w;
    cv::Mat mask_mat = mask.getMat();
    for (int i = 0; i < mask_mat.rows; i++) {
        if (cv::countNonZero(mask_mat.row(i) != 0) > 0) { // almeno uno zero
            if (first_row == -1)
                first_row = i;
        }
        else if (first_row != -1) {
            last_row = i;
            break;
        }
    }
    for (int j = 0; j < mask_mat.cols; j++) {
        if (cv::countNonZero(mask_mat.col(j) != 0) > 0) {
            if (first_col == -1)
                first_col = j;
        }
        else if (first_col != -1) {
            last_col = j;
            break;
        }
    }
    img = img(cv::Rect(first_col, first_row, last_col - first_col, last_row - first_row));
    //

    // run inference
    auto output_tensors = SuperPoint::runSession(img);
    //

    // get the output
    std::vector<float> keypoints_raw, scores_raw, descriptors_raw;
    parse_superpoint_output(output_tensors, keypoints_raw, scores_raw, descriptors_raw);

    cv::Mat descriptors_mat;
    superpoint_raw_to_cv(keypoints_raw, scores_raw, descriptors_raw, keypoints, descriptors_mat);
    descriptors_mat.copyTo(descriptors);

    for (int i = 0; i < keypoints.size(); i++) {
        keypoints[i].pt += cv::Point2f(first_col, first_row);
    }
    //

    //TODO mask keypoints
}

std::vector<Ort::Value> SuperPoint::runSession(cv::Mat img) {

    const char* input_names[1] = { "input" };
    const char* output_names[2] = { "scores", "descriptors" };

    auto input_type_info = session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
    auto input_dims = input_type_info.GetShape();

    cv::Mat img_float;
    img.convertTo(img_float, CV_32FC1, 1.0 / 255.0);

    int new_h = img.rows;
    int new_w = img.cols;

    std::vector<float> input_tensor_values(new_h * new_w);
    for (int r = 0; r < new_h; r++) {
        for (int c = 0; c < new_w; c++) {
            input_tensor_values[r * new_w + c] = img_float.at<float>(r, c);
        }
    }

    input_dims[2] = new_h;
    input_dims[3] = new_w;

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor_values.data(),
        input_tensor_values.size(),
        input_dims.data(),
        input_dims.size()
    );

    std::vector<Ort::Value> output_tensors = session->Run(Ort::RunOptions{ nullptr },
        input_names, &input_tensor, 1,
        output_names, 2);

    return output_tensors;
}