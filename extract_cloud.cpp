/***********************************************************
 * Author:  Daniel Tran
 *          Liam Gogley
 *
 * Purpose: The following .cpp file will utilize the Intel
 *          realsense camera to stream and capture frame
 *          data of the environment. Color is then applied
 *          and a point cloud is generated and saved to
 *          a point cloud data format (.pcd).
 *
 * Version 0.09 - Last Editted 11/07/18
 *
 * Rev:     Implementation of RGB Texture function to map
 *          color to point cloud data.
 *
 ***********************************************************/

#define _CRT_SECURE_NO_WARNINGS
#define _WINSOCK_DEPRECATED_NO_WARNINGS
#include <iostream>
#include <algorithm> 
#include <string>
#include <chrono>  

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/thread/thread.hpp>

// PCL Headers
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/crop_box.h>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <boost/thread/thread.hpp>
#include <pcl/io/io.h>
//#include <pcl/visualization/cloud_viewer.h>

#include <fstream>
#include <sstream>
#include <vector>

using namespace std;

typedef pcl::PointXYZRGB RGB_Cloud;
typedef pcl::PointCloud<RGB_Cloud> point_cloud;
typedef point_cloud::Ptr cloud_pointer;
typedef point_cloud::Ptr prevCloud;

// Prototypes
void Load_PCDFile(void);
bool userInput(void);
void cloudViewer(void);
void waitForMatlab(void);

// Global Variables
string cloudFile; // .pcd file name
string prevCloudFile; // .pcd file name (Old cloud)
int i = 1; // Index for incremental file name

int main() try
{

    //Load generated PCD file for viewing
    Load_PCDFile();

    cout << "Exiting Program... " << endl;
    return EXIT_SUCCESS;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}


void Load_PCDFile(void)
{
    string filePath = "20210312_3axis/";
    string outPath = "20210312_3axis/";
    string openFileName;

    // Generate object to store cloud in .pcd file
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudView(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudFiltered(new pcl::PointCloud<pcl::PointXYZRGB>);

    ifstream in(filePath + "point_cloud_pose.txt");
    vector<vector<double>> fields;
    if (in) {
        string line;

        while (getline(in, line)) {
            stringstream sep(line);
            string field;

            fields.push_back(vector<double>());

            while (getline(sep, field, ',')) {
                fields.back().push_back(stod(field));
            }
        }
    }

    //==========================
       // Pointcloud Visualization
       //==========================
       // Create viewer object titled "Captured Frame"
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Captured Frame"));
    // Set background of viewer to black
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addCoordinateSystem(0.2);
    for (int j = 0; j < fields.size(); j++)
    {
        openFileName = filePath + "Captured_Frame" + to_string(j+1) + ".ply";
        pcl::io::loadPLYFile(openFileName, *cloudView); // Load .pcd File

        vector<double> paras = fields[j];
        vector<double> pos = vector<double>(paras.begin(), paras.begin()+3);
        vector<double> euler = vector<double>(paras.begin() + 3, paras.begin() + 6);
        vector<double> quat = vector<double>(paras.begin() + 6, paras.begin() + 10);
        pcl::CropBox<pcl::PointXYZRGB> boxFilter;
        boxFilter.setMin(Eigen::Vector4f(-0.15, -0.15, -0.12, 1.0));
        boxFilter.setMax(Eigen::Vector4f(0.15, 0.15, 0.12, 1.0));
        boxFilter.setTranslation(Eigen::Vector3f(pos[0], pos[1], pos[2]));
        boxFilter.setRotation(Eigen::Vector3f(euler[2], euler[1], euler[0]));
        boxFilter.setInputCloud(cloudView);
        boxFilter.filter(*cloudFiltered);

        //Eigen::Affine3f transform = Eigen::Affine3f::Identity();
        //Eigen::Affine3f inverse_transform = Eigen::Affine3f::Identity();

        //pcl::getTransformation(pos[0], pos[1], pos[2]
        //    ,euler[2], euler[1], euler[0],
        //    transform);
        //inverse_transform = transform.inverse();

        //pcl::transformPointCloud(*cloudView, *cloudFiltered, inverse_transform);

        Eigen::Vector3f cubeTranslation(pos[0], pos[1], pos[2]);
        Eigen::Quaternionf cubeRot(quat[0], quat[1], quat[2], quat[3]);

        //Eigen::Vector3f cubeTranslation(0, 0, 0);
        //Eigen::Quaternionf cubeRot(1, 0, 0, 0);

      
        //// Add generated point cloud and identify with string "Cloud"
        //viewer->addPointCloud<pcl::PointXYZRGB>(cloudView, "Cloud"+to_string(j));
        //// Default size for rendered points
        //viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "Cloud" + to_string(j));
        //// Add a cube
        //viewer->addCube(cubeTranslation, cubeRot, 0.3, 0.3, 0.24);
        //viewer->setRepresentationToWireframeForAllActors();
        //// Viewer Properties
        //viewer->initCameraParameters();  // Camera Parameters for ease of viewing

        //cout << endl;
        //cout << "Press [Q] in viewer to continue. " << endl;

        //viewer->spin(); // Allow user to rotate point cloud and view it

        //viewer->removeAllShapes();
        //viewer->removeAllPointClouds();

        pcl::io::savePCDFileASCII(filePath + "Cropped_Frame" + to_string(j) + ".pcd", *cloudFiltered);

        cout << "Processed " << to_string(j) << endl;


        // Note: No method to close PC visualizer, pressing Q to continue software flow only solution.
    }


}

//========================================
// userInput
// - Prompts user for a char to 
// test for decision making.
// [y|Y] - Capture frame and save as .pcd
// [n|N] - Exit program
//========================================
bool userInput(void) {

    bool setLoopFlag;
    bool inputCheck = false;
    char takeFrame; // Utilize to trigger frame capture from key press ('t')
    do {

        // Prompt User to execute frame capture algorithm
        cout << endl;
        cout << "Generate a Point Cloud? [y/n] ";
        cin >> takeFrame;
        cout << endl;
        // Condition [Y] - Capture frame, store in PCL object and display
        if (takeFrame == 'y' || takeFrame == 'Y') {
            setLoopFlag = true;
            inputCheck = true;
            takeFrame = 0;
        }
        // Condition [N] - Exit Loop and close program
        else if (takeFrame == 'n' || takeFrame == 'N') {
            setLoopFlag = false;
            inputCheck = true;
            takeFrame = 0;
        }
        // Invalid Input, prompt user again.
        else {
            inputCheck = false;
            cout << "Invalid Input." << endl;
            takeFrame = 0;
        }
    } while (inputCheck == false);

    return setLoopFlag;
}