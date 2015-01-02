#ifndef BACKGROUND_SUBTRACT_H
#define BACKGROUND_SUBTRACT_H

#include "common.h"
#include "log.h"
#include "frame_reader.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.h>
#include <math.h>

void WriteData(Mat& matData)
{
    if(matData.empty())
    {
        cout<<"Mat empty!"<<endl;
        return;
    }
    for(int r=0; r<matData.rows; ++r)
    {
        for(int c=0; c<matData.cols; ++c)
        {
            cout<<matData.at<float>(r,c)<<'\t';
        }
        cout<<endl;
    }
}

Mat backgroundSubtract(const Mat& rawImage, Mat& dx, Mat& dy)
{
    const float mvTre = 8.0;
    const int gridStep = 16;
    int mb_height = dx.rows;
    int mb_width = dx.cols;
    Mat foregroundImage;
    Mat flag;

    flag = Mat::zeros(mb_height, mb_width, CV_32FC1);

    if(rawImage.channels() != 1)
        cvtColor(rawImage, foregroundImage, CV_BGR2GRAY);
    else
        foregroundImage = rawImage.clone();

    for(int blk_r = 0; blk_r < mb_height; ++blk_r)
    {
        for(int blk_c = 0; blk_c < mb_width; ++blk_c)
        {
            if(std::sqrt(dx.at<float>(blk_r,blk_c)*dx.at<float>(blk_r,blk_c)+dy.at<float>(blk_r,blk_c)*dy.at<float>(blk_r,blk_c)) < mvTre)
            {
                flag.at<float>(blk_r, blk_c) = 0;
//                for(int r = 0; r < gridStep; ++r)
//                {
//                    for(int c = 0; c < gridStep; ++c)
//                    {
//                        foregroundImage.at<u_int8_t>(blk_r*gridStep+r, blk_c*gridStep+c) = 0;
//                    }
//                }
            }
            else
            {
                flag.at<float>(blk_r, blk_c) = 1.0;
            }
        }
    }

    for(int blk_r = 1; blk_r < mb_height-1; ++blk_r)
    {
        for(int blk_c = 1; blk_c < mb_width-1; ++blk_c)
        {
            if(flag.at<u_int8_t>(blk_r, blk_c) != 0)
            {
                flag.at<float>(blk_r-1, blk_c) = 1.0;
                flag.at<float>(blk_r+1, blk_c) = 1.0;
                flag.at<float>(blk_r, blk_c-1) = 1.0;
                flag.at<float>(blk_r, blk_c+1) = 1.0;
                flag.at<float>(blk_r-1, blk_c-1) = 1.0;
                flag.at<float>(blk_r-1, blk_c+1) = 1.0;
                flag.at<float>(blk_r+1, blk_c-1) = 1.0;
                flag.at<float>(blk_r+1, blk_c+1) = 1.0;
            }
        }
    }

    for(int blk_r = 0; blk_r < mb_height; ++blk_r)
    {
        for(int blk_c = 0; blk_c < mb_width; ++blk_c)
        {
            if(flag.at<float>(blk_r,blk_c) == 0)
            {
                for(int r = 0; r < gridStep; ++r)
                {
                    for(int c = 0; c < gridStep; ++c)
                    {
                        foregroundImage.at<u_int8_t>(blk_r*gridStep+r, blk_c*gridStep+c) = 0;
                    }
                }
                dx.at<float>(blk_r,blk_c) = 0;
                dy.at<float>(blk_r,blk_c) = 0;
            }
        }
    }

    return foregroundImage;
}

#endif // BACKGROUND_SUBTRACT_H
