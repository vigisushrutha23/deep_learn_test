#include <iostream>
#include <sstream>
#include "include/tennis_train.h"



/**
 * @brief TennisTrainer::TennisTrainer
 * Constructor to intilise traininer with image directory name,training width "w"and height "" of images.
 * Also number of samples can be specified
 */
TennisTrainer::TennisTrainer(std::string directory,std::string labels,int w,int h,int samples)
{
    image_directory_=directory;
    labels_file_=labels;
    image_training_width_=w;
    image_training_height_=h;
    number_of_samples_=samples;
    convertAllImages(image_directory_,Scale,image_training_width_,image_training_height_,training_images_);
    readStoreLabels(labels_file_,training_labels_);
    trainTennisRacket();
}

TennisTrainer::~TennisTrainer(){}

/**
 * @brief TennisTrainer::convertOneImage
 * @param imagefilename
 * @param scale scaling pixel intensities from 0-255 to 0-1;
 * @param w width of image
 * @param h height of image
 * @param data complete data vector of the training images.
 * Function to convert one image into a suitable format for inputting into the CNN based on an example given on the tiny_dnn git page
 */
void TennisTrainer::convertOneImage(const std::string& imagefilename,
                                    double scale,
                                    int w,
                                    int h,
                                    std::vector<vec_t>& data)
{
    auto img = cv::imread(imagefilename, cv::IMREAD_GRAYSCALE);
    if (img.data == nullptr) return; // cannot open, or it's not an image

    cv::Mat_<uint8_t> resized;

    cv::resize(img, resized, cv::Size(w, h));
    vec_t d;

    std::transform(resized.begin(), resized.end(), std::back_inserter(d),
                   [=](uint8_t c) { return c * scale; });
    data.push_back(d);
}

/**
 * @brief TennisTrainer::convertAllImages
 * Loops through all the the files in a directory in a numerical order.
 */
void TennisTrainer::convertAllImages(const std::string& directory,double scale, int w,int h,std::vector<vec_t>& data)
{

    /*path dpath(directory);

        BOOST_FOREACH(const path& p,std::make_pair(directory_iterator(dpath), directory_iterator())) {
            if (is_directory(p)) continue;*/
    std::stringstream p;
    for(int i=1;i<=number_of_samples_;i++)
    {
        std::stringstream p;
        p<<image_directory_<<i<<".jpeg";
        //cout<<"\n"<<p.str();
        convertOneImage(p.str(), scale, w, h, data);

    }

}

/**
 * @brief TennisTrainer::readStoreLabels
 * @param filename name of the file containing labels
 * @param label_data container for all labels
 * Reads a text files of labels arranged according to the file names.
 */
void TennisTrainer::readStoreLabels(const std::string filename,vector<label_t>& label_data)
{
    std::fstream myfile(filename, std::ios_base::in);

    label_t label_value;
    while (myfile >> label_value)
    {
        //cout<<"\n"<<label_value;
        label_data.push_back(label_value);
    }

}

/**
 * @brief TennisTrainer::trainTennisRacket
 * Function to train and save the CNN based aain on an example on the tiny_dnn git page. The model made is just a first prototype
 */
void TennisTrainer::trainTennisRacket()
{
    network<sequential> nn;
    //creating layers. convulution->average pooling->convulution->pooling->convolution->fully connected
    nn << conv<tan_h>(32, 32, 5, 1, 6) // C1, 1@32x32-in, 6@28x28-out
       << ave_pool<tan_h>(28, 28, 6, 2) // S2, 6@28x28-in, 6@14x14-out
       << conv<tan_h>(14, 14, 5, 6, 16) // C3, 6@14x14-in, 16@10x10-in
       << ave_pool<tan_h>(10, 10, 16, 2) // S4, 16@10x10-in, 16@5x5-out
       << conv<tan_h>(5, 5, 5, 16, 120) // C5, 16@5x5-in, 120@1x1-out
       << fc<tan_h>(120, 2); // F6, 120-in, 2-out
    cout << "\n started learning \n" ;
    adagrad optimizer;

    nn.train<mse>(optimizer,training_images_, training_labels_, 1000, 1000); //training for 100 epochs
    nn.test(training_images_, training_labels_).print_detail(std::cout);;

    // save
    nn.save("tennis_net1"); // saving neural network TODO: Add user functionality for this later.
}

int main()
{
    cout << "Initialising Trainer" << endl;
    TennisTrainer first_train("../training_images/","../training_labels.txt",32,32,40);
    return 0;
}

