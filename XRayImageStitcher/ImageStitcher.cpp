
#define NOMINMAX
#include "ImageStitcher.hpp"
#include "dicomfile.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include "superglue/SuperPoint.hpp"
#include "superglue/SuperGlue.hpp"
#include <numeric>

cv::Mat ImageStitcher::dicom_to_cv_16(std::string path) {
    auto dicomfile = DicomFile();
    bool ret = dicomfile.OpenRead(path);
    if (!ret) {
        std::cerr << "Failed to openread" << std::endl;
        std::cerr << "Quitting..." << std::endl;
        exit(0);
    }

    int height = dicomfile.Height();
    int width = dicomfile.Width();

    auto ptr = dicomfile.GetFrame(1);

    cv::Mat img16(height, width, CV_16UC1);
    memcpy(img16.data, ptr, width * height * sizeof(uint16_t));
    //cv::Mat img16 = cv::Mat(height, width, CV_16UC1, (void*)ptr);

    dicomfile.Close();

    return img16;
}

cv::Mat ImageStitcher::cv16_to_cv8(cv::Mat img16) {
    cv::Mat img8;
    double minVal, maxVal;
    cv::minMaxLoc(img16, &minVal, &maxVal);

    img16.convertTo(img8, CV_8UC1, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));    
    //img16.convertTo(img8, CV_8UC1, 255.0 / 65535.0);

    return img8;
}

void ImageStitcher::newProcess()
{
	sources.clear();
    //TODO clear result and result_raw too
}

bool ImageStitcher::addImage(std::string dcmFilename, CroppingRect croppingRect)
{
	//TODO check if extension is .dcm and if file exists (?), else return false
    sources.push_back({ dcmFilename, nullptr, 0, 0, 0, 0, cv::Mat(), ImageData::Type::FILENAME, nullptr, croppingRect});
	return true;
}

bool ImageStitcher::addImage(void* rawPixels, int w, int h, int ch, int depth, CroppingRect croppingRect)
{
    sources.push_back({ "", rawPixels, w, h, ch, depth, cv::Mat(), ImageData::Type::RAW, nullptr, croppingRect});
    return true;
}

bool ImageStitcher::addImage(const cv::Mat& mat, CroppingRect croppingRect)
{
    sources.push_back({ "", nullptr, 0, 0, 0, 0, mat, ImageData::Type::CV_MAT, nullptr, croppingRect});
    return true;
}

bool ImageStitcher::addImageWithRaw(std::string dcmFilename, std::string dcmFilenamer, CroppingRect croppingRect)
{
    //TODO check if extension is .dcm and if file exists (?), else return false
    ImageData* rawImg = new ImageData { dcmFilenamer, nullptr, 0, 0, 0, 0, cv::Mat(), ImageData::Type::FILENAME, nullptr, croppingRect};
    sources.push_back({ dcmFilename, nullptr, 0, 0, 0, 0, cv::Mat(), ImageData::Type::FILENAME, rawImg, croppingRect});
    return true;
}

unsigned int ImageStitcher::getNumberOfImages() const
{
	return sources.size();
}

cv::Mat ImageStitcher::crop_image(const cv::Mat& img, const CroppingRect& crop) {
    //get x, y, width, height
    int x = crop.leftCrop;
    int y = crop.topCrop;
    int w = img.cols - crop.leftCrop - crop.rightCrop;
    int h = img.rows - crop.topCrop - crop.bottomCrop;

    //check unvalid values
    if (x < 0) 
        x = 0;
    if (y < 0)
        y = 0;
    if (w < 1) 
        w = 1;
    if (h < 1)
        h = 1;
    if (x + w > img.cols) w = img.cols - x;
    if (y + h > img.rows) h = img.rows - y;

    cv::Rect roi(x, y, w, h);

    return img(roi);
}

bool ImageStitcher::get_img_from_source(ImageData& imgData, cv::Mat& out_img_work, cv::Mat& out_img_res, cv::Mat& out_img_raw) {
    out_img_res = load_image(imgData);

    if (out_img_res.depth() == 0) //CV_8U 
        out_img_work = out_img_res;
    else if (out_img_res.depth() == 2) //CV_16U 
        out_img_work = cv16_to_cv8(out_img_res);
    else
        return false;

    if (imgData.raw != nullptr) {
        out_img_raw = load_image(*imgData.raw);
    }
    return true;
}

cv::Mat ImageStitcher::load_image(const ImageData& imgData) {
    cv::Mat out_img_res;
    switch (imgData.type) {
    case ImageData::Type::FILENAME:
        out_img_res = dicom_to_cv_16(imgData.imageFilename);
        break;
    case ImageData::Type::RAW:
        out_img_res = cv::Mat(imgData.height, imgData.width, CV_MAKETYPE(imgData.depth, imgData.channels), imgData.rawImageData);
        break;
    case ImageData::Type::CV_MAT:
        out_img_res = imgData.matrix.clone();
        break;
    }
    if (imgData.croppingRect.leftCrop || imgData.croppingRect.topCrop || imgData.croppingRect.rightCrop || imgData.croppingRect.bottomCrop)
        out_img_res = crop_image(out_img_res, imgData.croppingRect);
    return out_img_res;
}

int ImageStitcher::stitch(DetectorAlgorithm detectorAlgorithm, MatchingAlgorithm matchingAlgorithm, HomographyMethod homographyMethod)
{
    stitchDataVec.clear();

    //load first image
    cv::Mat img1, img1_res, img1_raw;
    if (!get_img_from_source(sources[0], img1, img1_res, img1_raw))
        return false;

    //raw result will be created only if all sources contains raw images
    bool use_raw = std::all_of(sources.begin(), sources.end(), [](auto src) { return src.raw != nullptr; });

    //cv::imwrite(outFilename + "_1_16.png", img1_res);
    //cv::imwrite(outFilename + "_1_8.png", img1);
    //

    cv::Mat gray_out, gray_out_raw;
    for (int i = 1; i < sources.size(); i++) {
        std::cout << "Step: " << i << "/" << sources.size()-1 << std::endl;

        //load next image
        cv::Mat img2, img2_res, img2_raw;
        if (!get_img_from_source(sources[i], img2, img2_res, img2_raw))
            return false;

        //std::string partialFilename = outFilename + "_" + std::to_string(i + 1);
        //std::string partialFilename = "_" + std::to_string(i + 1);
        //cv::imwrite(partialFilename + "_16.png", img2_res);
        //cv::imwrite(partialFilename + "_8.png", img2);

        //

        //create masks to select the image part of interest
        float part = portionToAnalyse;
        int rect_height = img2.rows * part;
        cv::Mat mask_bottomPart = cv::Mat::zeros(img1.size(), CV_8UC1);
        cv::Rect bottomPart(0, img1.rows - rect_height, img1.cols, rect_height);
        mask_bottomPart(bottomPart).setTo(255);

        cv::Mat mask_topPart = cv::Mat::zeros(img2.size(), CV_8UC1);
        cv::Rect topPart(0, 0, img2.cols, rect_height);
        mask_topPart(topPart).setTo(255);
        //

        //run key detection
        //L1 and L2 norms are preferable choices for SIFT and SURF descriptors,
        //NORM_HAMMING should be used with ORB, BRISK and BRIEF, 
        //NORM_HAMMING2 should be used with ORB when WTA_K == 3 or 4 (see ORB::ORB constructor description).
        //SIFT, SURF, KAZE use floating - point descriptors => use NORM_L2.
        //ORB, BRISK, AKAZE use binary descriptors => use NORM_HAMMING.
        cv::Ptr<cv::Feature2D> detector;
        int bfMatcherMethod;
        float ratio_thresh = 0.6f;
        DescriptorType descriptorType;
        switch (detectorAlgorithm) {
        case DetectorAlgorithm::SIFT:
            //detector = cv::SIFT::create(0, 3, 0.01, 10, 1.6, false); //WORKING
            detector = cv::SIFT::create(0, 3, 0.01, 10, 1.6, false);
            bfMatcherMethod = cv::NormTypes::NORM_L2;
            descriptorType = DescriptorType::FLOAT;
            break;

        case DetectorAlgorithm::ORB: //TODO FINDS A LOT OF KNN_MATCHES BUT THEN GOOD_MATCHES ARE LOW
            //detector = cv::ORB::create(500, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20); default
            detector = cv::ORB::create(5000, 1.2f, 8, 5, 0, 2, cv::ORB::FAST_SCORE, 31, 5);
            bfMatcherMethod = cv::NormTypes::NORM_HAMMING;
            descriptorType = DescriptorType::BINARY;
            break;

        /*case DetectorAlgorithm::SURF: //requires the extra module opencv_contrib
            detector = cv::xfeatures2d::SURF::create(400);
            bfMatcherMethod = cv::NormTypes::NORM_L2;
            descriptorType = DescriptorType::FLOAT;*/

        case DetectorAlgorithm::BRISK:
            //detector = cv::BRISK::create(6, 3, 1.2f); //SLOW 200k keypoints
            //detector = cv::BRISK::create(14, 3, 1.2f); //SLOW 100k keypoints
            detector = cv::BRISK::create(6, 3, 1.2f); //
            bfMatcherMethod = cv::NormTypes::NORM_HAMMING;
            descriptorType = DescriptorType::BINARY;
            break;

        case DetectorAlgorithm::AKAZE:
            detector = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, 0.00005f, 4, 4, cv::KAZE::DIFF_PM_G2, -1); //OPTIMAL
            bfMatcherMethod = cv::NormTypes::NORM_HAMMING;
            descriptorType = DescriptorType::BINARY;
            break;

        case DetectorAlgorithm::KAZE:
            detector = cv::KAZE::create(false, true, 0.00005f, 4, 4, cv::KAZE::DIFF_PM_G2); //OPTIMAL, SLOWER
            bfMatcherMethod = cv::NormTypes::NORM_L2;
            descriptorType = DescriptorType::FLOAT;
            break;

        case DetectorAlgorithm::SUPERPOINT:
            //detector = SuperPoint::create(superpoint_model_path, 0.00005f);
            detector = SuperPoint::create(superpoint_model_path, 0.0005f);
            bfMatcherMethod = cv::NormTypes::NORM_L2;
            descriptorType = DescriptorType::FLOAT;
            ratio_thresh = 0.85;
            break;
        }

        std::vector<cv::KeyPoint> kp1, kp2;
        cv::Mat desc1, desc2;
        if (detectorAlgorithm == DetectorAlgorithm::SUPERPOINT) {
            detector->detectAndCompute(img1, mask_bottomPart, kp1, desc1);
            detector->detectAndCompute(img2, mask_topPart, kp2, desc2);
        }
        else {
            detector->detect(img1, kp1, mask_bottomPart);
            detector->detect(img2, kp2, mask_topPart);
            int N = 30000;
            if (kp1.size() > N) {
                cv::KeyPointsFilter::retainBest(kp1, N);
            }
            if (kp2.size() > N) {
                cv::KeyPointsFilter::retainBest(kp2, N);
            }
            detector->compute(img1, kp1, desc1);
            detector->compute(img2, kp2, desc2);
        }
        //

        //// show keypoint images
        //cv::Mat out1;
        //cv::Mat img_color1;
        //cv::cvtColor(img1, img_color1, cv::COLOR_GRAY2BGR);
        //cv::drawKeypoints(img_color1, kp1, out1, cv::Scalar(0, 0, 255));
        //cv::imwrite("keypoints0.png", out1);
        //cv::Mat out2;
        //cv::Mat img_color2;
        //cv::cvtColor(img2, img_color2, cv::COLOR_GRAY2BGR);
        //cv::drawKeypoints(img_color2, kp2, out2, cv::Scalar(0, 0, 255));
        //cv::imwrite("keypoints1.png", out2);
        ////
        
        std::cout << "kp found: " << kp1.size() << ", " << kp2.size() << std::endl;

        //match the keypoints
        cv::Ptr<cv::DescriptorMatcher> matcher;
        int k = 2;
        switch (matchingAlgorithm) {
        case MatchingAlgorithm::BRUTEFORCE:
            matcher = cv::BFMatcher::create(bfMatcherMethod, false);
            break;

        case MatchingAlgorithm::FLANN: {
            cv::Ptr<cv::flann::SearchParams> searchParams = cv::makePtr<cv::flann::SearchParams>(50);
            if (descriptorType == DescriptorType::FLOAT) {
                cv::Ptr<cv::flann::IndexParams> indexParams1 = cv::makePtr<cv::flann::KDTreeIndexParams>(5);
                matcher = cv::makePtr<cv::FlannBasedMatcher>(indexParams1, searchParams);
            }
            else {
                cv::Ptr<cv::flann::IndexParams> indexParams2 = cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2);
                matcher = cv::makePtr<cv::FlannBasedMatcher>(indexParams2, searchParams);
            }
            break;
        }
        case MatchingAlgorithm::SUPERGLUE:
            //matcher = SuperGlue::create(superglue_model_path, 0.00002f);
            matcher = SuperGlue::create(superglue_model_path, 0.2f);
            matcher.dynamicCast<SuperGlue>()->init(
                kp1, kp2,
                img1.cols, rect_height,
                img2.cols, rect_height,
                0, img1.rows - rect_height);
            k = 1;
            break;
        }

        std::vector<std::vector<cv::DMatch>> knn_matches;
        try {
            matcher->knnMatch(desc1, desc2, knn_matches, k);
        }
        catch (...) {
            return 1;
        }
        std::cout << "knn_matches:  " << knn_matches.size() << std::endl;
        //

        //filter matches to find the good matches
        std::vector<cv::DMatch> good_matches;
        if (k == 2) {
            //Lowe's ratio test to find good matches (k=2)
            for (size_t i = 0; i < knn_matches.size(); i++) {
                if (knn_matches[i].size() == 2) {
                    if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
                        good_matches.push_back(knn_matches[i][0]);
                    }
                }
                else if (knn_matches[i].size() == 1) {
                    good_matches.push_back(knn_matches[i][0]);
                }
            }
        }
        else if (k == 1) {
            for (size_t i = 0; i < knn_matches.size(); i++) {
                if (knn_matches[i][0].distance < 0.8) //TODO is it a valid procedure?
                    good_matches.push_back(knn_matches[i][0]);
            }
        }
        std::cout << "good_matches: " << good_matches.size() << std::endl;
        //
        if (good_matches.size() < 4) {
            return 1;
        }

        //create an image to display the matches, just for visual purposes [TODO remove this]
        //cv::Mat img_matches;
        //cv::Mat img1_color, img2_color;
        //cv::cvtColor(img1, img1_color, cv::COLOR_GRAY2BGR);
        //cv::cvtColor(img2, img2_color, cv::COLOR_GRAY2BGR);
        //cv::vconcat(img1_color, img2_color, img_matches);
        //for (const auto& match : good_matches) {
        //    cv::Point2f pt1 = kp1[match.queryIdx].pt;
        //    cv::Point2f pt2 = kp2[match.trainIdx].pt + cv::Point2f(0, img1.rows);
        //    cv::circle(img_matches, pt1, 6, cv::Scalar(0, 255, 0), -1);
        //    cv::circle(img_matches, pt2, 6, cv::Scalar(0, 255, 0), -1);
        //    cv::line(img_matches, pt1, pt2, cv::Scalar(0, 0, 255), 4);
        //}
        //cv::imwrite(partialFilename + "_m.png", img_matches);
        //std::cout << partialFilename + "_m.png created!" << std::endl;
        //

        //calculate homography matrix and overlap
        std::vector<cv::Point2f> pts1, pts2;
        for (const auto& match : good_matches) {
            pts1.push_back(kp1[match.queryIdx].pt);
            pts2.push_back(kp2[match.trainIdx].pt);
        }

        cv::Mat mask;
        cv::Mat H = findHomography(pts2, pts1, (int)homographyMethod, 10.0, mask); //TODO try changing ransacReprojThreshold
        if (H.empty()) {
            std::cout << "Homography matrix is empty! Cannot perform perspective transform." << std::endl;
            return 1;
        }

        //std::cout << "Mat H:" << std::endl;
        //for (int i = 0; i < 3; i++) {
        //    for (int j = 0; j < 3; j++) {
        //        std::cout << H.at<double>(i, j) << ", ";
        //    }
        //    std::cout << std::endl;
        //}
        float dist = homographyDistanceFromTranslation(H);
        std::cout << "Dist: " << dist << std::endl;

        std::vector<cv::Point2f> corners_img2 = {
            cv::Point2f(0, 0),
            cv::Point2f(img2.cols, 0),
            cv::Point2f(0, img2.rows),
            cv::Point2f(img2.cols, img2.rows)
        };
        std::vector<cv::Point2f> corners_transformed;
        cv::perspectiveTransform(corners_img2, corners_transformed, H);

        float overlap = img1.rows - corners_transformed[0].y;
        std::cout << "overlap: " << overlap << std::endl;
        //

        //save result image
        //int final_height = std::max(corners_transformed[2].y, corners_transformed[3].y) + 1;
        //if (final_height < img1_res.rows) {
        //    std::cout << "final_height < img1_res.rows" << std::endl;
        //    return 1;
        //}
        ////int final_height = 2 * img1_res.rows;
        //cv::Size final_size(img1_res.cols, final_height);
        //cv::Mat stitched(final_size, img1_res.type(), cv::Scalar::all(0));
        //img1_res.copyTo(stitched(cv::Rect(0, 0, img1_res.cols, img1_res.rows)));
        //cv::warpPerspective(img2_res, stitched, H, final_size, cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);
        //cv::imwrite(partialFilename + "_warp.png", stitched);
        //

        //get result image with alpha blending
        if (apply_alpha_blending(img1_res, img2_res, gray_out, corners_transformed, overlap, H))
            return 1;
        if (use_raw) {
            if (apply_alpha_blending(img1_raw, img2_raw, gray_out_raw, corners_transformed, overlap, H))
                return 1;
        }
        //cv::imwrite(partialFilename + "_warp_a.png", gray_out);
        //

        //the next first image is the current result
        img1 = cv16_to_cv8(gray_out);
        img1_res = gray_out;
        img1_raw = gray_out_raw;
        //
        
        stitchDataVec.emplace_back(
            kp1.size(),
            kp2.size(),
            knn_matches.size(),
            good_matches.size(),
            dist,
            overlap
            );

        //TODO use corners_transformed to obtain the mask where to find the keypoints for the next image 
        // (otherwise there's a risk for them to be searched outside the image borders)
    }

    //cv::imwrite(outFilename + "_warp_a_final.png", gray_out);
    //if (use_raw) {
    //    cv::imwrite(outFilename + "_warp_a_final_raw.png", gray_out_raw);
    //}

    result = gray_out.clone();
    if (use_raw) {
        result_raw = gray_out_raw.clone();
    }

    std::cout << "Done!" << std::endl;

    return 0;
}

int ImageStitcher::apply_alpha_blending(const cv::Mat& img1_res, const cv::Mat& img2_res, cv::Mat& gray_out, std::vector<cv::Point2f> & corners_transformed, int overlap, cv::Mat& H) {
    int height = std::max(corners_transformed[2].y, corners_transformed[3].y) + 1;
    if (height < img1_res.rows) {
        std::cout << "final_height < img1_res.rows" << std::endl;
        return 1;
    }
    int start_y = std::min(corners_transformed[0].y, corners_transformed[1].y);
    cv::Size size(img1_res.cols, height);

    cv::Mat alpha2 = cv::Mat::ones(img2_res.size(), CV_8UC1) * 255;
    cv::Mat alpha2_big(size, CV_8UC1, cv::Scalar(0));
    cv::warpPerspective(alpha2, alpha2_big, H, size, cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);

    cv::Mat img2_ch0(size, img2_res.type(), cv::Scalar(0));
    cv::warpPerspective(img2_res, img2_ch0, H, size, cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);

    cv::Mat img1_ch0(size, img1_res.type());
    img1_res.copyTo(img1_ch0(cv::Rect(0, 0, img1_res.cols, img1_res.rows)));

    cv::Mat gray1f, gray2f, alpha2f;

    //cv::Mat alpha1_big(size, CV_8UC1, cv::Scalar(0)); //TODO create directly alpha1f
    //alpha1_big(cv::Rect(0, 0, img1_res.cols, img1_res.rows)).setTo(255);
    //alpha1_big.convertTo(alpha1f, CV_32F, 1.0 / 255.0);
    cv::Mat alpha1f(size, CV_32F, cv::Scalar(0.0f));
    alpha1f(cv::Rect(0, 0, img1_res.cols, img1_res.rows)).setTo(1.0f);

    float alpha_step = 255.0f / overlap;
    float alpha_value = 0;
    for (int r = 0; r < overlap; r++) {
        alpha2_big.row(start_y + r).setTo(alpha_value);
        alpha_value += alpha_step;
    }
    alpha2_big.convertTo(alpha2f, CV_32F, 1.0 / 255.0);

    img1_ch0.convertTo(gray1f, CV_32F, 1.0 / 65535.0);
    img2_ch0.convertTo(gray2f, CV_32F, 1.0 / 65535.0);

    //alpha compositing: https://en.wikipedia.org/wiki/Alpha_compositing
    //cv::Mat alpha_out = alpha2f + alpha1f.mul(1.0f - alpha2f); //alpha1f unnecessary since it is always 1
    //gray_out = (gray2f.mul(alpha2f) + gray1f.mul(alpha1f.mul(1.0f - alpha2f))).mul(alpha_out);
    cv::Mat alpha_out = alpha2f + (1.0f - alpha2f);
    gray_out = (gray2f.mul(alpha2f) + gray1f.mul(1.0f - alpha2f)).mul(alpha_out);
    gray_out.convertTo(gray_out, CV_16UC1, 65535.0);
    return 0;
}



//

float ImageStitcher::homographyDistanceFromTranslation(const cv::Mat& H)
{
    double tx = H.at<double>(0, 2);
    double ty = H.at<double>(1, 2);
    cv::Mat T = (cv::Mat_<double>(3, 3) <<
        1, 0, tx,
        0, 1, ty,
        0, 0, 1);
    return cv::norm(H - T, cv::NORM_L2);
}