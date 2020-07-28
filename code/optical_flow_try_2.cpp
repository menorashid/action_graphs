#include <iostream>
#include <fstream>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/gpu/gpu.hpp"

using namespace std;
using namespace cv;
using namespace cv::gpu;


inline bool isFlowCorrect(Point2f u)
{
    return !cvIsNaN(u.x) && !cvIsNaN(u.y) && fabs(u.x) < 1e9 && fabs(u.y) < 1e9;
}

static Vec3b computeColor(float fx, float fy)
{
    static bool first = true;

    // relative lengths of color transitions:
    // these are chosen based on perceptual similarity
    // (e.g. one can distinguish more shades between red and yellow
    //  than between yellow and green)
    const int RY = 15;
    const int YG = 6;
    const int GC = 4;
    const int CB = 11;
    const int BM = 13;
    const int MR = 6;
    const int NCOLS = RY + YG + GC + CB + BM + MR;
    static Vec3i colorWheel[NCOLS];

    if (first)
    {
        int k = 0;

        for (int i = 0; i < RY; ++i, ++k)
            colorWheel[k] = Vec3i(255, 255 * i / RY, 0);

        for (int i = 0; i < YG; ++i, ++k)
            colorWheel[k] = Vec3i(255 - 255 * i / YG, 255, 0);

        for (int i = 0; i < GC; ++i, ++k)
            colorWheel[k] = Vec3i(0, 255, 255 * i / GC);

        for (int i = 0; i < CB; ++i, ++k)
            colorWheel[k] = Vec3i(0, 255 - 255 * i / CB, 255);

        for (int i = 0; i < BM; ++i, ++k)
            colorWheel[k] = Vec3i(255 * i / BM, 0, 255);

        for (int i = 0; i < MR; ++i, ++k)
            colorWheel[k] = Vec3i(255, 0, 255 - 255 * i / MR);

        first = false;
    }

    const float rad = sqrt(fx * fx + fy * fy);
    const float a = atan2(-fy, -fx) / (float) CV_PI;

    const float fk = (a + 1.0f) / 2.0f * (NCOLS - 1);
    const int k0 = static_cast<int>(fk);
    const int k1 = (k0 + 1) % NCOLS;
    const float f = fk - k0;

    Vec3b pix;

    for (int b = 0; b < 3; b++)
    {
        const float col0 = colorWheel[k0][b] / 255.0f;
        const float col1 = colorWheel[k1][b] / 255.0f;

        float col = (1 - f) * col0 + f * col1;

        if (rad <= 1)
            col = 1 - rad * (1 - col); // increase saturation with radius
        else
            col *= .75; // out of range

        pix[2 - b] = static_cast<uchar>(255.0 * col);
    }

    return pix;
}

static void drawOpticalFlow(const Mat_<float>& flowx, const Mat_<float>& flowy, Mat& dst, float maxmotion = -1)
{
    dst.create(flowx.size(), CV_8UC3);
    dst.setTo(Scalar::all(0));

    // determine motion range:
    float maxrad = maxmotion;

    if (maxmotion <= 0)
    {
        maxrad = 1;
        for (int y = 0; y < flowx.rows; ++y)
        {
            for (int x = 0; x < flowx.cols; ++x)
            {
                Point2f u(flowx(y, x), flowy(y, x));

                if (!isFlowCorrect(u))
                    continue;

                maxrad = max(maxrad, sqrt(u.x * u.x + u.y * u.y));
            }
        }
    }

    for (int y = 0; y < flowx.rows; ++y)
    {
        for (int x = 0; x < flowx.cols; ++x)
        {
            Point2f u(flowx(y, x), flowy(y, x));

            if (isFlowCorrect(u))
                dst.at<Vec3b>(y, x) = computeColor(u.x / maxrad, u.y / maxrad);
        }
    }
}

static void showFlow(const char* name, const GpuMat& d_flowx, const GpuMat& d_flowy)
{
    Mat flowx(d_flowx);
    Mat flowy(d_flowy);

    Mat out;
    drawOpticalFlow(flowx, flowy, out, 10);

    imwrite("flow.jpg", out);
}

static void convertFlowToImage( const Mat& flowIn, float lowerBound, float higherBound, string out_file) 
{   
    // Mat flowIn(d_flowx);
    Mat flowOut(flowIn.rows, flowIn.cols, CV_8UC1);

    #define CAST(v, L, H) ((v) > (H) ? 255 : (v) < (L) ? 0 : cvRound(255*((v) - (L))/((H)-(L))))
    for (int i = 0; i < flowIn.rows; ++i) {
        for (int j = 0; j < flowIn.cols; ++j) {
            float x = flowIn.at<float>(i,j);
            flowOut.at<uchar>(i,j) = CAST(x, lowerBound, higherBound);
        }
    }
    #undef CAST

    imwrite(out_file, flowOut);
}


// static void get_flow(OpticalFlowDual_TVL1_GPU tvl1, string im1, string im2, string dir_u, string dir_v){
//     // cout <<"im1 "<< im1<<endl;
//     // cout <<"im2 "<< im2<<endl;
//     // cout <<"dir_u "<< dir_u<<endl;
//     // cout <<"dir_v "<< dir_v<<endl;
    
//     string out_file_u(dir_u);
//     out_file_u.append(im1, im1.length() - 16, im1.length());

//     string out_file_v(dir_v);
//     out_file_v.append(im1, im1.length() - 16, im1.length());

//     cout <<"out_file_u "<< out_file_u<<endl;
//     cout <<"out_file_v "<< out_file_v<<endl;

//     Mat frame0 = imread(im1, IMREAD_GRAYSCALE);
//     Mat frame1 = imread(im2, IMREAD_GRAYSCALE);

//     GpuMat d_frame0(frame0);
//     GpuMat d_frame1(frame1);

//     GpuMat d_flowx(frame0.size(), CV_32FC1);
//     GpuMat d_flowy(frame0.size(), CV_32FC1);

//     tvl1(d_frame0, d_frame1, d_flowx, d_flowy);

//     convertFlowToImage( d_flowx, -20., 20., out_file_u);
//     convertFlowToImage( d_flowy, -20., 20., out_file_v);
// }


int main(int argc, const char* argv[])
{   

    cv::gpu::setDevice(1);
    vector<string> list;
    ifstream file(argv[1]);
    string str; 
    while (getline(file, str))
    {
        list.push_back(str);
    }

    cout << list.size() << endl;

    OpticalFlowDual_TVL1_GPU tvl1;
    // cout <<"helloooo" <<endl;

    string dir_u = argv[2];
    string dir_v = argv[3];

    cout <<list[0]<<endl;
    cout <<list[1]<<endl;

    cout <<dir_u<<endl;
    cout <<dir_v<<endl;

    Mat frame0= imread(list[0], IMREAD_GRAYSCALE);
    // cout <<"helloooo" <<endl;
    Mat frame1= imread(list[1], IMREAD_GRAYSCALE);
    // cout <<"helloooo" <<endl;
    int rows = frame0.rows;
    int cols = frame0.cols;
    // cout <<"helloooo" <<endl;
    GpuMat d_frame0(rows, cols, CV_32FC1);
    // (frame0);
    GpuMat d_frame1(rows, cols, CV_32FC1);
    // (frame1);

    GpuMat d_flowx(rows, cols, CV_32FC1);
    GpuMat d_flowy(rows, cols, CV_32FC1);
    Mat flowIn_u (rows, cols, CV_32FC1);
    Mat flowIn_v (rows, cols, CV_32FC1);

    // cout <<"helloooo" <<endl;
    for (int i =0;i<list.size()-1;i++){

        // cout <<i <<endl;

        // get_flow( tvl1, list[i], list[i+1], argv[2], argv[3]);

        string im1 = list[i];
        string im2 = list[i+1];

        string out_file_u(dir_u);
        out_file_u.append(im1, im1.length() - 16, im1.length());

        string out_file_v(dir_v);
        out_file_v.append(im1, im1.length() - 16, im1.length());

        if (i>0){
            frame1.copyTo(frame0);
            // imread(im1, IMREAD_GRAYSCALE);
            frame1 =  imread(im2, IMREAD_GRAYSCALE);
        }
        // if (i==0){
        //     d_frame0 = 
        // }
        d_frame0.upload(frame0);
        d_frame1.upload(frame1);
        tvl1(d_frame0, d_frame1, d_flowx, d_flowy);
        d_flowx.download(flowIn_u);
        d_flowy.download(flowIn_v);

        convertFlowToImage( flowIn_u, -20., 20., out_file_u);
        convertFlowToImage( flowIn_v, -20., 20., out_file_v);

    }
    cout << list.size() << endl;



    // if (argc < 3)
    // {
    //     cerr << "Usage : " << argv[0] << "<frame0> <frame1>" << endl;
    //     return -1;
    // }

    // Mat frame0 = imread(argv[1], IMREAD_GRAYSCALE);
    // Mat frame1 = imread(argv[2], IMREAD_GRAYSCALE);

    // cout << "helloooo" << endl;

    // if (frame0.empty())
    // {
    //     cerr << "Can't open image ["  << argv[1] << "]" << endl;
    //     return -1;
    // }
    // if (frame1.empty())
    // {
    //     cerr << "Can't open image ["  << argv[2] << "]" << endl;
    //     return -1;
    // }

    // if (frame1.size() != frame0.size())
    // {
    //     cerr << "Images should be of equal sizes" << endl;
    //     return -1;
    // }

    // GpuMat d_frame0(frame0);
    // GpuMat d_frame1(frame1);

    // GpuMat d_flowx(frame0.size(), CV_32FC1);
    // GpuMat d_flowy(frame0.size(), CV_32FC1);

    // OpticalFlowDual_TVL1_GPU tvl1;
    
    // cout << "before" << endl;

    // tvl1(d_frame0, d_frame1, d_flowx, d_flowy);

    // cout << "after" << endl;

    // convertFlowToImage( d_flowx, -20, 20, "../scratch/checking_flo_u.jpg");
    // convertFlowToImage( d_flowy, -20, 20, "../scratch/checking_flo_v.jpg");
    
    // cout << "done" << endl;        

    return 0;
}