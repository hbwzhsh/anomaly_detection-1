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

//        imshow("frame.RawImage", frame.RawImage);
//        waitKey(0);
//        imshow("curRawImageGray", curRawImageGray);
//        waitKey(0);

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
            frame.rsd = Mat::zeros(curRawImageGray.rows/dctGridStep, curRawImageGray.cols/dctGridStep, CV_32FC1);
            return;
        }

        residual = Mat::zeros(curRawImageGray.rows/dctGridStep, curRawImageGray.cols/dctGridStep, CV_32FC1);
        residualFrame = Mat::zeros(preRawImageGray.rows, preRawImageGray.cols, CV_32FC1);

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
                                = float(curRawImageGray.at<u_int8_t>(next_blk_j*gridStep+j, next_blk_i*gridStep+i) - preRawImageGray.at<u_int8_t>(blk_j*gridStep+j, blk_i*gridStep+i));
                    }
                }
            }
        }

        for(int blk_j = 0; blk_j < curRawImageGray.rows/dctGridStep; ++blk_j)
        {
            for(int blk_i = 0; blk_i < curRawImageGray.cols/dctGridStep; ++blk_i)
            {
                Mat block(dctGridStep, dctGridStep, CV_32FC1);
                Mat trans_block(dctGridStep, dctGridStep, CV_32FC1);
                Mat dct_block(dctGridStep, dctGridStep, CV_32FC1);

                for(int j = 0; j < dctGridStep; ++j)
                    for(int i = 0; i < dctGridStep; ++i)
                        block.at<float>(j, i) = residualFrame.at<float>(blk_j*dctGridStep+j, blk_i*dctGridStep+i);

                dct(block, trans_block);
                dct_block = Quantize(trans_block);

                for(int j = 0; j < dctGridStep; ++j)
                    for(int i = 0; i < dctGridStep; ++i)
                        residualFrame.at<float>(blk_j*dctGridStep+j, blk_i*dctGridStep+i) = dct_block.at<float>(j, i);
            }
        }

        for(int blk_j = 0; blk_j < curRawImageGray.rows/dctGridStep; ++blk_j)
        {
            for(int blk_i = 0; blk_i < curRawImageGray.cols/dctGridStep; ++blk_i)
            {
                residual.at<float>(blk_j, blk_i) = residualFrame.at<float>(blk_j*dctGridStep+0, blk_i*dctGridStep+0);
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
