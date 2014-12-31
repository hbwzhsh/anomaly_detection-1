#ifndef RESIDUAL_H
#define RESIDUAL_H

#include "common.h"
#include "log.h"
#include "frame_reader.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.h>
#include <math.h>

struct Residual
{
    static const float resThr = 5.0;
    static const float absThr = 50.0;
    static const int gridStep = 16;
    bool firstFlag;
    Frame preFrame;
    Frame curFrame;
    Mat preRawImageGray;
    Mat curRawImageGray;
    Mat residualFrame;
    Mat foregroundFrame;
    Mat backgroundFrame;
    Mat_<float> foregroundDx, foregroundDy;
    Mat_<float> backgroundDx, backgroundDy;

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

        if(firstFlag == true)
        {
            preFrame = curFrame = frame;
            preRawImageGray = curRawImageGray.clone();
            firstFlag = false;
            frame.foregroundFrame = Mat::zeros(curRawImageGray.rows, curRawImageGray.cols, CV_32FC1);
            frame.backgroundFrame = Mat::zeros(curRawImageGray.rows, curRawImageGray.cols, CV_32FC1);
            frame.foregroundDx = Mat_<float>::zeros(frame.Dx.rows, frame.Dx.cols);
            frame.foregroundDy = Mat_<float>::zeros(frame.Dx.rows, frame.Dx.cols);
            frame.backgroundDx = Mat_<float>::zeros(frame.Dx.rows, frame.Dx.cols);
            frame.backgroundDy = Mat_<float>::zeros(frame.Dx.rows, frame.Dx.cols);
            return;
        }

        residualFrame = Mat::zeros(curRawImageGray.rows, curRawImageGray.cols, CV_32FC1);
        foregroundFrame = Mat::zeros(curRawImageGray.rows, curRawImageGray.cols, CV_32FC1);
        backgroundFrame = Mat::zeros(curRawImageGray.rows, curRawImageGray.cols, CV_32FC1);
        foregroundDx = Mat_<float>::zeros(frame.Dx.rows, frame.Dx.cols);
        foregroundDy = Mat_<float>::zeros(frame.Dx.rows, frame.Dx.cols);
        backgroundDx = Mat_<float>::zeros(frame.Dx.rows, frame.Dx.cols);
        backgroundDy = Mat_<float>::zeros(frame.Dx.rows, frame.Dx.cols);

        for(int blk_j = 0; blk_j < preFrame.Dx.rows; ++blk_j)
        {
            for(int blk_i = 0; blk_i < preFrame.Dy.cols; ++blk_i)
            {
                int next_blk_j = (blk_j*gridStep + gridStep/2 + (preFrame.Dy)(blk_j, blk_i))/gridStep;
                int next_blk_i = (blk_i*gridStep + gridStep/2 + (preFrame.Dx)(blk_j, blk_i))/gridStep;
                next_blk_j = max(0, min(next_blk_j, preFrame.Dx.rows));
                next_blk_i = max(0, min(next_blk_i, preFrame.Dx.cols));

                for(int j = 0; j < gridStep; ++j)
                {
                    for(int i = 0; i < gridStep; ++i)
                    {
                        residualFrame.at<float>(blk_j*gridStep+j, blk_i*gridStep+i)
                                = float(curRawImageGray.at<u_int8_t>(next_blk_j*gridStep+j, next_blk_i*gridStep+i)
                                    - preRawImageGray.at<u_int8_t>(blk_j*gridStep+j, blk_i*gridStep+i));
                    }
                }
            }
        }

        for(int j = 0; j < residualFrame.rows; ++j)
        {
            for(int i = 0; i < residualFrame.cols; ++i)
            {
                if(residualFrame.at<float>(j, i) < resThr)
                {
                    foregroundFrame.at<float>(j, i) = 0;
                    backgroundFrame.at<float>(j, i) = residualFrame.at<float>(j, i);
                }
                else
                {
                    foregroundFrame.at<float>(j, i) = residualFrame.at<float>(j, i);
                    backgroundFrame.at<float>(j, i) = 0;
                }
            }
        }

        for(int blk_j = 0; blk_j < preFrame.Dx.rows; ++blk_j)
        {
            for(int blk_i = 0; blk_i < preFrame.Dy.cols; ++blk_i)
            {
                float sum = 0;
                for(int j = 0; j < gridStep; ++j)
                {
                    for(int i = 0; i < gridStep; ++i)
                    {
                        sum += (residualFrame.at<float>(blk_j*gridStep+j, blk_i*gridStep+i) > 0.0 ?
                                    residualFrame.at<float>(blk_j*gridStep+j, blk_i*gridStep+i) : -residualFrame.at<float>(blk_j*gridStep+j, blk_i*gridStep+i));
                    }
                }
                if(sum/(gridStep*gridStep) < absThr)
                {
//                    frame.Dx(blk_j, blk_i) = 0;
//                    frame.Dy(blk_j, blk_i) = 0;
                    foregroundDx(blk_j, blk_i) = 0;
                    foregroundDy(blk_j, blk_i) = 0;
                    backgroundDx(blk_j, blk_i) = preFrame.Dx(blk_j, blk_i);
                    backgroundDy(blk_j, blk_i) = preFrame.Dy(blk_j, blk_i);
                }
                else
                {
                    foregroundDx(blk_j, blk_i) = preFrame.Dx(blk_j, blk_i);
                    foregroundDy(blk_j, blk_i) = preFrame.Dy(blk_j, blk_i);
                    backgroundDx(blk_j, blk_i) = 0;
                    backgroundDy(blk_j, blk_i) = 0;
                }
            }
        }

//        frame.foregroundFrame = foregroundFrame.clone();
//        frame.backgroundFrame = backgroundFrame.clone();
        frame.foregroundDx = foregroundDx.clone();
        frame.foregroundDy = foregroundDy.clone();
        frame.backgroundDx = backgroundDx.clone();
        frame.backgroundDy = backgroundDy.clone();

        preFrame = curFrame;
        preRawImageGray = curRawImageGray.clone();
    }
};

#endif // RESIDUAL_H
