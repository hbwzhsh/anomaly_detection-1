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
    bool firstFlag;
    Frame preFrame;
    Frame curFrame;
    Mat preRawImageGray;
    Mat curRawImageGray;
    Mat preGradientX, preGradientY;
    Mat curGradientX, curGradientY;
    Mat Gx, Gy;

    Residual() : firstFlag(true)
    {
    }

    void Update(Frame& frame)
    {
        if(frame.RawImage.empty())
        {
            return;
        }

        if(frame.RawImage.channels() != 1)
        {
            cvtColor(frame.RawImage, curRawImageGray, CV_BGR2GRAY);
        }
        else
        {
            curRawImageGray = frame.RawImage;
        }

        if(firstFlag == true)
        {
            preFrame = curFrame = frame;
            preRawImageGray = curRawImageGray;
            firstFlag = false;
//            frame.rsd = Mat::zeros(curRawImageGray.rows, curRawImageGray.cols, CV_32FC1);
            frame.Gx = Mat::zeros(curRawImageGray.rows, curRawImageGray.cols, CV_32FC1);
            frame.Gy = Mat::zeros(curRawImageGray.rows, curRawImageGray.cols, CV_32FC1);
            return;
        }

        curFrame = frame;

        Sobel(preRawImageGray, preGradientX, CV_32FC1, 1, 0, 1);
        Sobel(preRawImageGray, preGradientY, CV_32FC1, 0, 1, 1);
        Sobel(curRawImageGray, curGradientX, CV_32FC1, 1, 0, 1);
        Sobel(curRawImageGray, curGradientY, CV_32FC1, 0, 1, 1);

        subtract(curGradientX, preGradientX, Gx);
        subtract(curGradientY, preGradientY, Gy);

        frame.Gx = Gx;
        frame.Gy = Gy;

        preFrame = curFrame;
        preRawImageGray = curRawImageGray;
    }
};

#endif // RESIDUAL_H
