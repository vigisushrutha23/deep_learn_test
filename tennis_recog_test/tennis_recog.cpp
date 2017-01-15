/**
  Program to do a simple recognition based on a pre-trained neural network.
  **/
#include <iostream>
#include <sstream>
#include "include/tennis_train.h"



//Function to convert one mage to suitable input format for the netork
void convertOneImage(const std::string& imagefilename,
                     double scale,
                     int w,
                     int h,
                     vec_t& data)
{
    auto img = cv::imread(imagefilename, cv::IMREAD_GRAYSCALE);
    if (img.data == nullptr) {cout<<"Invalid filename. Re run the program with the correct filename";return;} // cannot open, or it's not an image

    cv::Mat_<uint8_t> resized;

    cv::resize(img, resized, cv::Size(w, h));


    std::transform(resized.begin(), resized.end(), std::back_inserter(data),
                   [=](uint8_t c) { return c * scale; });

}

int main()
{
    string filename;
    cout<<"Enter Image Filename"; //getting filename from user
    cin>>filename;
    vec_t cnn_input;
    convertOneImage(filename,Scale,32,32,cnn_input);
    cout<<" Loading Network";
    network<sequential> nn;
    nn.load("tennis_net1");
    label_t prediction=nn.predict_label(cnn_input);
    if(int(prediction))
        cout<<"\nPreiction : Tennis racket is present in this image";
    else
        cout<<"\n Prediction : No tennis racket in this image";
    return 0;
}
