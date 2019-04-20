/*!
 * @author    Mayank Mittal
 * @email     mittalma@ethz.ch
 */
// C++
#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
// OpenCV
#include<opencv2/core/core.hpp>
// Boost
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
// ORBSLAM
#include<System.h>

using namespace std;

/*!
 * @brief Function to read data for TUM RGBD Dataset
 * @note Images present in the `rgb.txt` file are read
 * @param strPathToSequence string path to the directory containing the TUM-RGBD sequence
 * @param vstrImageFilenames output list of RGB files present in the directory
 * @param vTimestamps output list of timestamps of each RGB image
 */
void loadImagesTUM(const string& strPathToSequence, vector<string>& vstrImageFilenames, vector<double>& vTimestamps);

/*!
 * @brief Function to read data for KITTI Dataset
 * @note Images present in the directory `image_0` are read
 * @param sequencePath string path to the directory containing the KITTI sequence
 * @param vstrImageFilenames output list of RGB files present in the directory
 * @param vTimestamps output list of timestamps of each RGB image
 */
void loadImagesKITTI(const string& strPathToSequence, vector<string>& vstrImageFilenames, vector<double>& vTimestamps);

int main(int argc, char** argv) {
  // Definitions
  namespace po = boost::program_options;

  // Set defaults for modifiable configurations
  bool verbose = false;
  bool visualize = false;
  std::string sequenceType;
  std::string vocabularyPath;
  std::string settingsPath;
  std::string sequencePath;
  std::string outputPath;

  // Add options
  po::options_description desc("Options");
  desc.add_options()
    ("help,h", "print help messages")
    ("sequenceType,t", po::value<std::string>(&sequenceType)->required(), "set the type of sequence (kitti/tum)")
    ("vocabularyPath,f", po::value<std::string>(&vocabularyPath)->required(), "set the path of vocabulary")
    ("settingsPath,s", po::value<std::string>(&settingsPath)->required(), "set the path of the settings of the camera")
    ("sequencePath,i", po::value<std::string>(&sequencePath)->required(), "set the path to input sequence")
    ("outputPath,o", po::value<std::string>(&outputPath)->default_value(""), "set the path to store output")
    ("verbose,v", po::bool_switch(&verbose), "set verbosity")
    ("visualize,z", po::bool_switch(&visualize), "set visualization");

  // Override the loaded configurations if options provided
  po::variables_map results;
  try {
    po::store(po::parse_command_line(argc, argv, desc), results); // can throw

    if (results.count("help")) {
      cerr << "Usage: " << argv[0] << std::endl << desc;
      return 1;
    } else if (results.count("sequenceType")) {
      sequenceType = results["sequenceType"].as<std::string>();
      boost::algorithm::to_lower(sequenceType);
    } else if (results.count("vocabularyPath")) {
      vocabularyPath = results["vocabularyPath"].as<std::string>();
    } else if (results.count("settingsPath")) {
      settingsPath = results["settingsPath"].as<std::string>();
    } else if (results.count("sequencePath")) {
      sequencePath = results["sequencePath"].as<std::string>();
    } else if (results.count("outputPath")) {
      outputPath = results["outputPath"].as<std::string>();
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
  vector<string> vstrImageFilenames;
  vector<double> vTimestamps;
  if (sequenceType == "tum") {
    loadImagesTUM(sequencePath, vstrImageFilenames, vTimestamps);
  } else if (sequenceType == "kitti") {
    loadImagesKITTI(sequencePath, vstrImageFilenames, vTimestamps);
  } else {
    cerr << "Invalid sequence type provided!" << endl;
    return 1;
  }
  int nImages = vstrImageFilenames.size();

  // Create SLAM system. It initializes all system threads and gets ready to process frames.
  ORB_SLAM2::System SLAM(vocabularyPath, settingsPath, ORB_SLAM2::System::MONOCULAR, visualize);

  // Vector for tracking time statistics
  vector<float> vTimesTrack;
  vTimesTrack.resize(nImages);

  cout << endl << "-------" << endl;
  cout << "Start processing sequence ..." << endl;
  cout << "Images in the sequence: " << nImages << endl << endl;

  // Main loop
  cv::Mat im;
  for (int ni = 0; ni < nImages; ni++) {
    // Read image from file
    im = cv::imread(vstrImageFilenames[ni], CV_LOAD_IMAGE_UNCHANGED);
    double tframe = vTimestamps[ni];

    if (im.empty()) {
      cerr << endl << "Failed to load image at: "
           << string(argv[3]) << "/" << vstrImageFilenames[ni] << endl;
      return 1;
    }

#ifdef COMPILEDWITHC11
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
    std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

    // Pass the image to the SLAM system
    SLAM.TrackMonocular(im, tframe);

#ifdef COMPILEDWITHC11
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
    std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

    double ttrack = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

    vTimesTrack[ni] = ttrack;

    // Wait to load the next frame
    double T = 0;
    if (ni < nImages - 1) {
      T = vTimestamps[ni + 1] - tframe;
    } else if (ni > 0) {
      T = tframe - vTimestamps[ni - 1];
    }
    if (ttrack < T) {
      usleep((T - ttrack) * 1e6);
    }
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
  SLAM.SaveKeyFrameTrajectoryTUM(string(outputPath) + "/stamped_traj_estimate.txt");

  return 0;
}

void loadImagesTUM(const string& strPathToSequence, vector<string>& vstrImageFilenames, vector<double>& vTimestamps) {
  ifstream f;
  string strFile = strPathToSequence + "/rgb.txt";
  f.open(strFile.c_str());

  // skip first three lines
  string s0;
  getline(f, s0);
  getline(f, s0);
  getline(f, s0);

  while (!f.eof()) {
    string s;
    getline(f, s);
    if (!s.empty()) {
      stringstream ss;
      ss << s;
      double t;
      string sRGB;
      ss >> t;
      vTimestamps.push_back(t);
      ss >> sRGB;
      vstrImageFilenames.push_back(strPathToSequence + '/' + sRGB);
    }
  }
}

void loadImagesKITTI(const string& strPathToSequence, vector<string>& vstrImageFilenames, vector<double>& vTimestamps) {
  ifstream fTimes;
  string strPathTimeFile = strPathToSequence + "/image_00/times.txt";
  fTimes.open(strPathTimeFile.c_str());
  while (!fTimes.eof()) {
    string s;
    getline(fTimes, s);
    if (!s.empty()) {
      stringstream ss;
      ss << s;
      double t;
      ss >> t;
      vTimestamps.push_back(t);
    }
  }

  string strPrefixLeft = strPathToSequence + "/image_00/data/";

  const int nTimes = vTimestamps.size();
  vstrImageFilenames.resize(nTimes);

  for (int i = 0; i < nTimes; i++) {
    stringstream ss;
    ss << setfill('0') << setw(10) << i;
    vstrImageFilenames[i] = strPrefixLeft + ss.str() + ".png";
  }
}
