

#include "ImageStitcher.hpp"
#include <direct.h>
#include <fstream>
#include "json.hpp"



std::map<std::string, std::vector<std::vector<std::string>>> d = {
    //{ "0",  { { "a", "b", "c" }, { "d", "e", "f" } } },
    { "1",  { { "a", "b", "c", "d" } } },
    { "2",  { { "a", "b", "c", "d" } } },
    { "3",  { { "a", "b" }, { "c", "d" } } },
    { "4",  { { "a", "b", "c", "d" } } },
    { "5",  { { "a", "b" }, { "c", "d" } } },
    { "6",  { { "a", "b", "c", "d" } } },
    { "7",  { { "a", "b", "c", "d" } } },
    { "8",  { { "a", "b", "c", "d" } } },
    { "9",  { { "a", "b", "c", "d" } } },
    { "10", { { "a", "b", "c", "d" } } },
    { "11", { { "a", "b", "c", "d" } } },
    { "12", { { "c", "d" }, { "e", "f" } } },
    { "13", { { "a", "b", "c", "d" } } },
    { "14", { { "a", "b", "c", "d" } } },
    { "15", { { "a", "b", "c", "d" } } },
    { "16", { { "a", "b", "c" }, { "d", "e", "f" } } },
    { "17", { { "a", "b" }, { "c", "d" } } }
};


//int main() {
//     
//    ImageStitcher stitcher;
//
//    //_chdir(R"(E:\TESI_ARCHIVE\test_folder\p1)");
//    //stitcher.newProcess();
//    //stitcher.addImage("a.dcm");
//    //stitcher.addImage("b.dcm");
//    //stitcher.addImage("c.dcm");
//    //stitcher.addImage("d.dcm");
//
//    //stitcher.stitch(
//    //    "abcd",
//    //    ImageStitcher::DetectorAlgorithm::SIFT,
//    //    ImageStitcher::MatchingAlgorithm::FLANN,
//    //    ImageStitcher::HomographyMethod::RANSAC);
//
//    _chdir(R"(E:\TESI_ARCHIVE\test_folder\)");
//    std::string errored = "";
//    for (const auto& pair : d) {
//        const auto& list_of_lists = pair.second;
//
//        _chdir(("p" + pair.first).c_str());
//
//        std::cout << pair.first << std::endl;
//
//        for (const auto& sublist : list_of_lists) {
//            stitcher.newProcess();
//            std::cout << "new process: ";
//            std::string name = "";
//            for (const auto& val : sublist) {
//                stitcher.addImage(val + ".dcm");
//                std::cout << val << ", ";
//                name += val;
//            }
//            std::cout << std::endl;
//
//            try {
//                if (stitcher.stitch(name,
//                    ImageStitcher::DetectorAlgorithm::KAZE,
//                    ImageStitcher::MatchingAlgorithm::FLANN,
//                    ImageStitcher::HomographyMethod::PROSAC))
//                    throw std::runtime_error("error in the process");
//                cv::imwrite(R"(E:\TESI_ARCHIVE\test_folder\results\SUPERPOINT+SUPERGLUE+LSQ\)" + pair.first + "_" + name + ".png", stitcher.getResult());
//            }
//            catch (...) {
//                std::cout << "error" << std::endl;
//                errored += "\n" + pair.first + "_" + name;
//            }
//
//        }
//        _chdir("..");
//
//        
//
//    }
//
//    std::cout << std::endl << errored << std::endl;
//
//    return 0;
//}


//taken from https://docs.opencv.org/4.x/dd/d3d/tutorial_gpu_basics_similarity.html
//and modified to work with 16 bit grayscale images
double getMSSIM_16Grayscale(const cv::Mat& i1, const cv::Mat& i2)
{
    //const double C1 = 6.5025, C2 = 58.5225;
    const double C1 = pow((0.01 * (pow(2, 16)-1)), 2);
    const double C2 = pow((0.03 * (pow(2, 16)-1)), 2);
    /***************************** INITS **********************************/
    int d = CV_32F;

    cv::Mat I1, I2;
    i1.convertTo(I1, d);           // cannot calculate on one byte large values
    i2.convertTo(I2, d);

    cv::Mat I2_2 = I2.mul(I2);        // I2^2
    cv::Mat I1_2 = I1.mul(I1);        // I1^2
    cv::Mat I1_I2 = I1.mul(I2);        // I1 * I2

    /*************************** END INITS **********************************/

    cv::Mat mu1, mu2;   // PRELIMINARY COMPUTING
    GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
    GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);

    cv::Mat mu1_2 = mu1.mul(mu1);
    cv::Mat mu2_2 = mu2.mul(mu2);
    cv::Mat mu1_mu2 = mu1.mul(mu2);

    cv::Mat sigma1_2, sigma2_2, sigma12;

    GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;

    GaussianBlur(I2_2, sigma2_2, cv::Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;

    GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    cv::Mat t1, t2, t3;

    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

    cv::Mat ssim_map;
    divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;

    cv::Scalar mssim = mean(ssim_map); // mssim = average of ssim map
    return mssim[0];
}

double getPSNR_16Grayscale(const cv::Mat& I1, const cv::Mat& I2)
{
    cv::Mat s1;
    absdiff(I1, I2, s1);       // |I1 - I2|
    s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
    s1 = s1.mul(s1);           // |I1 - I2|^2

    cv::Scalar s = sum(s1);         // sum elements per channel

    double sse = s[0]; // sum channels

    if (sse <= 1e-10) // for small values return zero
        return 0;
    else
    {
        double mse = sse / (double)I1.total();
        double psnr = 10.0 * log10((65535.0 * 65535.0) / mse);
        return psnr;
    }
}

double getMSE_16Grayscale(const cv::Mat& I1, const cv::Mat& I2)
{
    cv::Mat s1;
    absdiff(I1, I2, s1);       // |I1 - I2|
    s1.convertTo(s1, CV_32F);  // cannot make a square on 16 bits
    s1 = s1.mul(s1);           // |I1 - I2|^2

    cv::Scalar s = cv::sum(s1);         // sum elements per channel

    double mse = s[0] / (double)I1.total();
    return mse;
}

int main() {

    ImageStitcher stitcher;

    //_chdir(R"(E:\TESI_ARCHIVE\test_folder\p1)");
    //stitcher.newProcess();
    //stitcher.addImage("a.dcm", ImageStitcher::CroppingRect(520, 0, 650, 0)); //p3
    //stitcher.addImage("b.dcm", ImageStitcher::CroppingRect(520, 0, 650, 0));
    //stitcher.addImage("c.dcm", ImageStitcher::CroppingRect(520, 0, 650, 0));
    //stitcher.addImage("d.dcm", ImageStitcher::CroppingRect(520, 0, 650, 0));
    //stitcher.addImage("a.dcm", ImageStitcher::CroppingRect(550, 0, 620, 0)); //p5
    //stitcher.addImage("b.dcm", ImageStitcher::CroppingRect(550, 0, 620, 0));
    //stitcher.addImage("c.dcm", ImageStitcher::CroppingRect(550, 0, 620, 0));
    //stitcher.addImage("d.dcm", ImageStitcher::CroppingRect(550, 0, 620, 0));
    //stitcher.addImage("a.dcm");
    //stitcher.addImage("b.dcm");
    //stitcher.addImage("c.dcm");
    //stitcher.addImage("d.dcm");

    //stitcher.stitch(
    //    ImageStitcher::DetectorAlgorithm::SUPERPOINT,
    //    ImageStitcher::MatchingAlgorithm::SUPERGLUE,
    //    ImageStitcher::HomographyMethod::RANSAC);
    //
    //cv::Mat res = stitcher.getResult();
    //cv::imwrite("result.png", res);
    //
    //return 0;

    std::vector<ImageStitcher::DetectorAlgorithm> detAll = {
        ImageStitcher::DetectorAlgorithm::SIFT,
        ImageStitcher::DetectorAlgorithm::ORB,
        ImageStitcher::DetectorAlgorithm::BRISK,
        ImageStitcher::DetectorAlgorithm::AKAZE,
        ImageStitcher::DetectorAlgorithm::KAZE,
        ImageStitcher::DetectorAlgorithm::SUPERPOINT
    };

    std::string detAllS[] = {
        "SIFT",
        "ORB",
        "BRISK",
        "AKAZE",
        "KAZE",
        "SUPERPOINT"
    };

    std::vector<ImageStitcher::MatchingAlgorithm> matAll = {
        ImageStitcher::MatchingAlgorithm::BRUTEFORCE,
        ImageStitcher::MatchingAlgorithm::FLANN,
        ImageStitcher::MatchingAlgorithm::SUPERGLUE,
    };

    std::string matAllS[] = {
        "BRUTEFORCE",
        "FLANN",
        "SUPERGLUE",
    };

    std::vector<ImageStitcher::HomographyMethod> homAll = {
        //ImageStitcher::HomographyMethod::LSQ,
        ImageStitcher::HomographyMethod::RANSAC,
        ImageStitcher::HomographyMethod::LMEDS,
        ImageStitcher::HomographyMethod::PROSAC,
    };

    std::string homAllS[] = {
        //"LSQ",
        "RANSAC",
        "LMEDS",
        "PROSAC",
    };

    //for (int det = 0; det < detAll.size(); det++) {
    //    for (int mat = 0; mat < matAll.size(); mat++) {
    //        for (int hom = 0; hom < homAll.size(); hom++) {
    //            _chdir(R"(E:\TESI_ARCHIVE\test_folder\comp_p3)");
    //            std::string name = detAllS[det] + "_" + matAllS[mat] + "_" + homAllS[hom];

    //            std::ifstream f((name+".png").c_str());
    //            if (f.good())
    //                continue;

    //            _chdir(R"(E:\TESI_ARCHIVE\test_folder\p3)");

    //            std::cout << "detector: " + detAllS[det] + ", matcher: " + matAllS[mat] + ", homog: " + homAllS[hom] << std::endl;


    //            int result = stitcher.stitch(
    //                name,
    //                detAll[det],
    //                matAll[mat],
    //                homAll[hom]);
    //            if (result != 0) {
    //                std::cout << "ERROR WITH THIS COMBINATION" << std::endl;
    //                continue;
    //            }
    //            cv::Mat res = stitcher.getResult();

    //            _chdir(R"(E:\TESI_ARCHIVE\test_folder\comp_p3)");
    //            cv::imwrite(name + ".png", res);

    //            //cv::Mat res1 = cv::imread("reference.png", cv::IMREAD_GRAYSCALE);

    //            //int maxRows = std::max(res1.rows, res2.rows);
    //            //int maxCols = std::max(res1.cols, res2.cols);

    //            //cv::Mat res1_padded, res2_padded;
    //            //cv::copyMakeBorder(res1, res1_padded,
    //            //    0, maxRows - res1.rows,
    //            //    0, maxCols - res1.cols,
    //            //    cv::BORDER_CONSTANT, cv::Scalar(0));

    //            //cv::copyMakeBorder(res2, res2_padded,
    //            //    0, maxRows - res2.rows,
    //            //    0, maxCols - res2.cols,
    //            //    cv::BORDER_CONSTANT, cv::Scalar(0));

    //            //cv::imwrite("p_ref" + std::to_string(det) + std::to_string(match) + ".png", res1_padded);
    //            //cv::imwrite("p_res" + std::to_string(det) + std::to_string(match) + ".png", res2_padded);

    //            //double ssim = getMSSIM(res1_padded, res2_padded)[0];

    //            //std::cout << "SSIM " << det << match << ": " << ssim << std::endl;
    //            //outFile << std::to_string(det) + std::to_string(match) << ": " << ssim << std::endl;
    //        }
    //    }
    //    
    //}


    _chdir(R"(E:\TESI_ARCHIVE\test_folder\split_test_m)");
    stitcher.newProcess();
    stitcher.addImage(cv::imread("s1.png", cv::IMREAD_UNCHANGED));
    stitcher.addImage(cv::imread("s2.png", cv::IMREAD_UNCHANGED));
    stitcher.addImage(cv::imread("s3.png", cv::IMREAD_UNCHANGED));
    stitcher.addImage(cv::imread("s4.png", cv::IMREAD_UNCHANGED));

    cv::Mat ref = cv::imread("reconstructed.png", cv::IMREAD_UNCHANGED);

    for (int det = 0; det < detAll.size(); det++) {
        for (int mat = 0; mat < matAll.size(); mat++) {
            for (int hom = 0; hom < homAll.size(); hom++) {

                if (matAllS[mat] == "SUPERGLUE" && detAllS[det] != "SUPERPOINT")
                    continue;

                std::string name = "__" + detAllS[det] + "_" + matAllS[mat] + "_" + homAllS[hom];

                std::ifstream f((name+".png").c_str());
                if (f.good()) {
                    std::cout << "Skipping: " << name << std::endl;
                    continue;
                }

                std::cout << "detector: " + detAllS[det] + ", matcher: " + matAllS[mat] + ", homog: " + homAllS[hom] << std::endl;

                cv::Mat res;
                int result = stitcher.stitch(
                    detAll[det],
                    matAll[mat],
                    homAll[hom]);
                if (result) {
                    std::cout << "ERROR WITH THIS COMBINATION" << std::endl;
                }
                else {
                    res = stitcher.getResult();
                    cv::imwrite(name + ".png", res);
                }
                
                auto stitchDataVec = stitcher.getStitchDataVec();
                using nlohmann::json;
                json j;

                // array dei dati
                j["stitches"] = json::array();
                for (const auto& s : stitchDataVec) {
                    j["stitches"].push_back({
                        {"nkp1",       s.nkp1},
                        {"nkp2",       s.nkp2},
                        {"knnMatches", s.knnMatches},
                        {"goodMatches",s.goodMatches},
                        {"hDist",      s.hDist},
                        {"overlap",    s.overlap}
                        });
                }

                j["nstitches"] = stitchDataVec.size();

                if (!result) {
                    cv::Mat res_resized;
                    cv::resize(res, res_resized, cv::Size(ref.cols, ref.rows));

                    j["metrics"] = {
                        {"MSE",   getMSE_16Grayscale(ref, res_resized)},
                        {"PSNR",  getPSNR_16Grayscale(ref, res_resized)},
                        {"MSSIM", getMSSIM_16Grayscale(ref, res_resized)}
                    };
                }
                else {
                    j["metrics"] = {};
                }

                std::ofstream outFile(name + "_stitchData.json");
                outFile << j.dump(4);

            }
        }
    }
    return 0;
    
    _chdir(R"(E:\TESI_ARCHIVE\test_folder\)");
    std::string errored = "";
    for (const auto& pair : d) {
        const auto& list_of_lists = pair.second;

        _chdir(("p" + pair.first).c_str());

        std::cout << pair.first << std::endl;

        for (const auto& sublist : list_of_lists) {
            stitcher.newProcess();
            std::cout << "new process: ";
            std::string name_ = "";
            for (const auto& val : sublist) {
                stitcher.addImage(val + ".dcm");
                std::cout << val << ", ";
                name_ += val;
            }
            std::cout << std::endl;

            //cv::Mat ref = cv::imread("reconstructed.png", cv::IMREAD_UNCHANGED);

            for (int det = 0; det < detAll.size(); det++) {
                for (int mat = 0; mat < matAll.size(); mat++) {
                    for (int hom = 0; hom < homAll.size(); hom++) {

                        if (matAllS[mat] == "SUPERGLUE" && detAllS[det] != "SUPERPOINT")
                            continue;

                        std::string name = name_ + "_" + detAllS[det] + "_" + matAllS[mat] + "_" + homAllS[hom];

                        std::ifstream f((name + "_stitchData.json").c_str());
                        if (f.good()) {
                            std::cout << "Skipping: " << name << std::endl;
                            continue;
                        }

                        std::cout << "detector: " + detAllS[det] + ", matcher: " + matAllS[mat] + ", homog: " + homAllS[hom] << std::endl;

                        cv::Mat res;
                        int result = 1;
                        try {
                            result = stitcher.stitch(
                                detAll[det],
                                matAll[mat],
                                homAll[hom]);
                        }
                        catch (...) {

                        }
                        if (result) {
                            std::cout << "ERROR WITH THIS COMBINATION" << std::endl;
                        }
                        else {
                            res = stitcher.getResult();
                            cv::imwrite(name + ".png", res);
                        }

                        auto stitchDataVec = stitcher.getStitchDataVec();
                        using nlohmann::json;
                        json j;

                        // array dei dati
                        j["stitches"] = json::array();
                        for (const auto& s : stitchDataVec) {
                            j["stitches"].push_back({
                                {"nkp1",       s.nkp1},
                                {"nkp2",       s.nkp2},
                                {"knnMatches", s.knnMatches},
                                {"goodMatches",s.goodMatches},
                                {"hDist",      s.hDist},
                                {"overlap",    s.overlap}
                                });
                        }

                        j["nstitches"] = stitchDataVec.size();

                        //if (!result) {
                        //    cv::Mat res_resized;
                        //    cv::resize(res, res_resized, cv::Size(ref.cols, ref.rows));

                        //    j["metrics"] = {
                        //        {"MSE",   getMSE_16Grayscale(ref, res_resized)},
                        //        {"PSNR",  getPSNR_16Grayscale(ref, res_resized)},
                        //        {"MSSIM", getMSSIM_16Grayscale(ref, res_resized)}
                        //    };
                        //}
                        //else {
                        //    j["metrics"] = {};
                        //}

                        std::ofstream outFile(name + "_stitchData.json");
                        outFile << j.dump(4);

                    }
                }
            }
        }
        _chdir("..");
    }

    std::cout << std::endl << errored << std::endl;    

    return 0;
}