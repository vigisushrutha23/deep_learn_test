#ifndef TENNIS_TRAIN_H
#define TENNIS_TRAIN_H
#define Scale 0.003921
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <boost/foreach.hpp>
#include <fstream>
#include "tiny_dnn/tiny_dnn.h"

using namespace std;
using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;


class TennisTrainer
{
public:
    TennisTrainer(std::string directory,std::string labels,int w,int h,int samples); //COnsturctor to initialise trainer
    ~TennisTrainer();
    void convertOneImage(const std::string& imagefilename,double scale,int w,int h,std::vector<vec_t>& data);// Function to convert one image into suitable input
    void convertAllImages(const std::string& directory,double scale, int w,int h,std::vector<vec_t>& data); //Function to convert multiple images in a directory and to store in proper format
    void readStoreLabels(const std::string filename,vector<label_t>& label_data); //Function to read and store labels
    void trainTennisRacket(); // Function which actually trains the network
private:
    vector<vec_t> training_images_; //vector to store all training images arranged in a row-wise indexed vector
    vector<label_t> training_labels_; // vector to store all training labels
    int number_of_samples_;
    string image_directory_;
    string labels_file_;
    int image_training_width_;
    int image_training_height_;


};

#endif // TENNIS_TRAIN_H
