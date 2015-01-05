#ifndef RESIDUAL_H
#define RESIDUAL_H

#include "common.h"
#include "log.h"
#include "frame_reader.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.h>

struct Residual
{
    static const int gridStep = 16;
    static const int dctGridStep = 8;
    bool firstFlag;
    Frame preFrame;
    Frame curFrame;
    Mat preRawImageGray;
    Mat curRawImageGray;
    Mat residualFrame;
    Mat residual;

    Residual() : firstFlag(true)
    {
    }

    void Update(Frame& frame)
    {
        if(frame.RawImage.empty())
                    return;

        if(frame.RawImage.channels() != 1)
            cvtColor(frame.RawImage, curRawImageGray, CV_BGR2GRAY);
        else
            curRawImageGray = frame.RawImage.clone();

        curFrame = frame;
        curFrame.Dx = frame.Dx.clone();
        curFrame.Dy = frame.Dy.clone();

        if(firstFlag == true)
        {
            preFrame = curFrame = frame;
            curFrame.Dx = frame.Dx.clone();
            curFrame.Dy = frame.Dy.clone();
            preFrame.Dx = curFrame.Dx.clone();
            preFrame.Dy = curFrame.Dy.clone();
            preRawImageGray = curRawImageGray.clone();
            firstFlag = false;
            frame.rsd = Mat::zeros(curRawImageGray.rows/8, curRawImageGray.cols/8, CV_32FC1);
            return;
        }

        residual = Mat::zeros(curRawImageGray.rows/8, curRawImageGray.cols/8, CV_32FC1);
        residualFrame = Mat::zeros(preRawImageGray.rows, preRawImageGray.cols, CV_32FC1);

        subtract(curRawImageGray, preRawImageGray, residualFrame);

        for(int blk_j = 0; blk_j < curRawImageGray.rows/8; ++blk_j)
        {
            for(int blk_i = 0; blk_i < curRawImageGray.cols/8; ++blk_i)
            {
                int zeroNum = 0;
                for(int j = 0; j < 8; ++j)
                {
                    for(int i = 0; i < 8; ++i)
                    {
                        if(residualFrame.at<float>(blk_j*8+j, blk_i*8+i) == 0)
                        {
                            ++zeroNum;
                        }
                    }
                }
                residual.at<float>(blk_j, blk_i) = (float(zeroNum))/(float(8*8));
            }
        }

        frame.rsd = residual.clone();

        preFrame = curFrame;
        preFrame.Dx = curFrame.Dx.clone();
        preFrame.Dy = curFrame.Dy.clone();
        preRawImageGray = curRawImageGray.clone();
    }

    Mat Quantize(const Mat& src)
    {
        Mat dst(src.rows, src.cols, CV_32FC1);
        for(int j = 0; j < src.rows; ++j)
        {
            for(int i = 0; i < src.cols; ++i)
            {
//                if(src.at<float>(j, i) < 2.0)
//                    dst.at<u_int8_t>(j, i) = 0;
//                else if(src.at<float>(j, i) > 127.0)
//                    dst.at<u_int8_t>(j, i) = 127;
//                else
//                    dst.at<u_int8_t>(j, i) = int16_t(src.at<float>(j, i));
                dst.at<float>(j, i) = round(src.at<float>(j, i));
            }
        }
        return dst;
    }
};

#endif // RESIDUAL_H
