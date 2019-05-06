/*!
 * @author    Mayank Mittal
 * @email     mittalma@ethz.ch
 */
// C++
#include<iostream>
#include<algorithm>
#include<fstream>
#include<iomanip>
#include<chrono>
// OpenCV
#include<opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
// Boost
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
// ORBSLAM
#include<System.h>

using namespace std;

/*!
 * @brief load images from the directory path provided
 * @note Images present in the `rgb.txt` file are read
 * @param strPathLeft string path to the directory containing the left camera images
 * @param strPathRight string path to the directory containing the right camera images
 * @param vstrImageLeft output list of RGB files present in the left camera's directory
 * @param vstrImageRight output list of RGB files present in the right camera's directory
 * @param vTimeStamps output list of timestamps of each RGB image
 */
void loadStereoImages(const string& strPathLeft, const string& strPathRight,
                vector<string>& vstrImageLeft, vector<string>& vstrImageRight,
                vector<double>& vTimeStamps);

/*!
 * @brief convert RGB/RGBA images to grayscale
 * @param input input image
 * @param output output grayscale image
 * @param mbRGB flag to whether image is RGB or BGR
 */
void convertImageToGray(const cv::Mat& input, cv::Mat& output, const bool& mbRGB);

int main(int argc, char** argv) {
  // Definitions
  namespace po = boost::program_options;

  // Set defaults for modifiable configurations
  bool verbose = false;
  bool visualize = false;
  std::string sequenceRightPath, sequenceLeftPath;
  std::string vocabularyPath;
  std::string settingsPath;
  std::string sequencePath;
  std::string outputFilePath;

  // Add options
  po::options_description desc("Options");
  desc.add_options()
    ("help,h", "print help messages")
    ("vocabularyPath,f", po::value<std::string>(&vocabularyPath)->required(), "set the path of vocabulary")
    ("settingsPath,s", po::value<std::string>(&settingsPath)->required(), "set the path of the settings of the camera")
    ("sequenceRightPath,r", po::value<std::string>(&sequenceRightPath)->required(), "set the path to input sequence with right camera")
    ("sequenceLeftPath,l", po::value<std::string>(&sequenceLeftPath)->required(), "set the path to input sequence with left camera")
    ("outputFilePath,o", po::value<std::string>(&outputFilePath)->default_value(""), "set the file path to store output trajectory")
    ("verbose,v", po::bool_switch(&verbose), "set verbosity")
    ("visualize,z", po::bool_switch(&visualize), "set visualization");

  // Override the loaded configurations if options provided
  po::variables_map results;
  try {
    po::store(po::parse_command_line(argc, argv, desc), results); // can throw

    if (results.count("help")) {
      cerr << "Usage: " << argv[0] << std::endl << desc;
      return 1;
    } else if (results.count("sequenceRightPath")) {
      sequenceRightPath = results["sequenceRightPath"].as<std::string>();
      boost::algorithm::to_lower(sequenceRightPath);
    } else if (results.count("sequenceLeftPath")) {
      sequenceLeftPath = results["sequenceLeftPath"].as<std::string>();
      boost::algorithm::to_lower(sequenceLeftPath);
    }else if (results.count("vocabularyPath")) {
      vocabularyPath = results["vocabularyPath"].as<std::string>();
    } else if (results.count("settingsPath")) {
      settingsPath = results["settingsPath"].as<std::string>();
    } else if (results.count("outputFilePath")) {
      outputFilePath = results["outputFilePath"].as<std::string>();
    } else if (results.count("verbose")) {
      verbose = results["verbose"].as<bool>();
    } else if (results.count("visualize")) {
      visualize = results["visualize"].as<bool>();
    }
    po::notify(results);
    // catch error if any
  } catch (po::error& e) {
    cerr << e.what() << std::endl << desc;
    return 1;
  }

  // Retrieve paths to images
  vector<string> vstrImageLeft;
  vector<string> vstrImageRight;
  vector<double> vTimeStamp;
  loadStereoImages(sequenceLeftPath, sequenceRightPath, vstrImageLeft, vstrImageRight, vTimeStamp);

  if (vstrImageLeft.empty() || vstrImageRight.empty()) {
    cerr << "ERROR: No images in provided path." << endl;
    return 1;
  }

  if (vstrImageLeft.size() != vstrImageRight.size()) {
    cerr << "ERROR: Different number of left and right images." << endl;
    return 1;
  }

  // Read rectification parameters
  cv::FileStorage fsSettings(settingsPath, cv::FileStorage::READ);
  if (!fsSettings.isOpened()) {
    cerr << "ERROR: Wrong path to settings" << endl;
    return -1;
  }

  cv::Mat K_l, K_r, P_l, P_r, R_l, R_r, D_l, D_r;
  fsSettings["LEFT.K"] >> K_l;
  fsSettings["RIGHT.K"] >> K_r;

  fsSettings["LEFT.P"] >> P_l;
  fsSettings["RIGHT.P"] >> P_r;

  fsSettings["LEFT.R"] >> R_l;
  fsSettings["RIGHT.R"] >> R_r;

  fsSettings["LEFT.D"] >> D_l;
  fsSettings["RIGHT.D"] >> D_r;

  int rows_l = fsSettings["LEFT.height"];
  int cols_l = fsSettings["LEFT.width"];
  int rows_r = fsSettings["RIGHT.height"];
  int cols_r = fsSettings["RIGHT.width"];
  int mbRGB = fsSettings["Camera.RGB"];

  if (K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() || R_l.empty() || R_r.empty() || D_l.empty() || D_r.empty() ||
      rows_l == 0 || rows_r == 0 || cols_l == 0 || cols_r == 0) {
    cerr << "ERROR: Calibration parameters to rectify stereo are missing!" << endl;
    return -1;
  }

  cv::Mat M1l, M2l, M1r, M2r;
  cv::initUndistortRectifyMap(K_l, D_l, R_l, P_l.rowRange(0, 3).colRange(0, 3), cv::Size(cols_l, rows_l), CV_32F, M1l, M2l);
  cv::initUndistortRectifyMap(K_r, D_r, R_r, P_r.rowRange(0, 3).colRange(0, 3), cv::Size(cols_r, rows_r), CV_32F, M1r, M2r);


  const int nImages = vstrImageLeft.size();

  // Create SLAM system. It initializes all system threads and gets ready to process frames.
  ORB_SLAM2::System SLAM(vocabularyPath, settingsPath, ORB_SLAM2::System::STEREO, visualize);

  // Vector for tracking time statistics
  vector<float> vTimesTrack;
  vTimesTrack.resize(nImages);

  cout << endl << "-------" << endl;
  cout << "Start processing sequence ..." << endl;
  cout << "Images in the sequence: " << nImages << endl << endl;

  // Main loop
  cv::Mat imLeft, imRight, imLeftRect, imRightRect;
  for (int ni = 0; ni < nImages; ni++) {
    // Read left and right images from file
    imLeft = cv::imread(vstrImageLeft[ni], CV_LOAD_IMAGE_UNCHANGED);
    imRight = cv::imread(vstrImageRight[ni], CV_LOAD_IMAGE_UNCHANGED);

    if (imLeft.empty()) {
      cerr << endl << "Failed to load image at: "
           << string(vstrImageLeft[ni]) << endl;
      return 1;
    }

    if (imRight.empty()) {
      cerr << endl << "Failed to load image at: "
           << string(vstrImageRight[ni]) << endl;
      return 1;
    }

    cv::remap(imLeft, imLeftRect, M1l, M2l, cv::INTER_LINEAR);
    cv::remap(imRight, imRightRect, M1r, M2r, cv::INTER_LINEAR);

    double tframe = vTimeStamp[ni];

    cv::Mat imLeftGray, imRightGray;
    imLeftGray = cv::Mat(imLeftRect.size(), imLeftRect.type());
    imRightGray = cv::Mat(imRightRect.size(), imRightRect.type());
    convertImageToGray(imLeftRect, imLeftGray, mbRGB);
    convertImageToGray(imRightRect, imRightGray, mbRGB);

    cerr << "Tracking frames at timestamp:" << tframe << endl;

#ifdef COMPILEDWITHC11
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
    std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

    // Pass the images to the SLAM system
    SLAM.TrackStereo(imLeftGray, imRightGray, tframe);

#ifdef COMPILEDWITHC11
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
    std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

    double ttrack = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

    vTimesTrack[ni] = ttrack;

    // Wait to load the next frame
    double T = 0;
    if (ni < nImages - 1)
      T = vTimeStamp[ni + 1] - tframe;
    else if (ni > 0)
      T = tframe - vTimeStamp[ni - 1];

    if (ttrack < T)
      usleep((T - ttrack) * 1e6);
  }

  // Stop all threads
  SLAM.Shutdown();

  // Tracking time statistics
  sort(vTimesTrack.begin(), vTimesTrack.end());
  float totaltime = 0;
  for (int ni = 0; ni < nImages; ni++) {
    totaltime += vTimesTrack[ni];
  }
  cout << "-------" << endl << endl;
  cout << "median tracking time: " << vTimesTrack[nImages / 2] << endl;
  cout << "mean tracking time: " << totaltime / nImages << endl;

  // Save camera trajectory
  SLAM.SaveTrajectoryTUM(outputFilePath);

  return 0;
}

void loadStereoImages(const string& strPathLeft, const string& strPathRight,
                vector<string>& vstrImageLeft, vector<string>& vstrImageRight,
                vector<double>& vTimeStamps) {
  ifstream fLeft, fRight;
  string strPathLeftFile = strPathLeft + "/rgb.txt";
  string strPathRightFile = strPathLeft + "/rgb.txt";

  fLeft.open(strPathLeftFile.c_str());
  fRight.open(strPathRightFile.c_str());

  // skip first three lines
  string s0, s1;
  getline(fLeft, s0);
  getline(fLeft, s0);
  getline(fLeft, s0);
  getline(fRight, s1);
  getline(fRight, s1);
  getline(fRight, s1);

  while (!fLeft.eof() || !fRight.eof()) {
    string s_left, s_right;
    getline(fLeft, s_left);
    getline(fRight, s_right);
    if (!s_left.empty()) {
      stringstream ss;
      ss << s_left;
      double t;
      string sRGB;
      ss >> t;
      vTimeStamps.push_back(t);
      ss >> sRGB;
      sRGB = strPathLeft + "/" + sRGB;
      vstrImageLeft.push_back(sRGB);
    }
    if (!s_right.empty()) {
      stringstream ss;
      ss << s_left;
      double t;
      string sRGB;
      ss >> t;
      ss >> sRGB;
      sRGB = strPathRight + "/" + sRGB;
      vstrImageRight.push_back(sRGB);
    }
  }
}

void convertImageToGray(const cv::Mat& input, cv::Mat& output, const bool& mbRGB) {

  if (input.channels() == 3) {
    if (mbRGB) {
      cv::cvtColor(input, output, CV_RGB2GRAY);
    } else {
      cv::cvtColor(input, output, CV_BGR2GRAY);
    }
  } else if (input.channels() == 4) {
    if (mbRGB) {
      cv::cvtColor(input, output, CV_RGBA2GRAY);
    } else {
      cv::cvtColor(input, output, CV_BGRA2GRAY);
    }
  }
}
