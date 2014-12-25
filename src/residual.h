#ifndef RESIDUAL_H
#define RESIDUAL_H

#include "common.h"
#include "log.h"
#include "frame_reader.h"
#include <cmath>
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
        curFrame = frame;

        int mb_width = curRawImageGray.cols/gridStep;
        int mb_height = curRawImageGray.rows/gridStep;

        if(firstFlag == true)
        {
            preFrame = curFrame = frame;
            preRawImageGray = curRawImageGray;
            firstFlag = false;
            frame.rsd = Mat::zeros(preRawImageGray.rows, preRawImageGray.cols, CV_32FC1);
            return;
        }

        residual = Mat::zeros(preRawImageGray.rows, preRawImageGray.cols, CV_32FC1);
        residualFrame = Mat::zeros(preRawImageGray.rows, preRawImageGray.cols, CV_32FC1);

        for(vector<MotionVector>::const_iterator iter = frame.MvInfo.begin(); iter != frame.MvInfo.end(); ++iter)
        {
            MotionVector mv = *iter;
            if(mv.NoMotionVector())
                continue;

            if(mv.SegmCode == '+')
            {
                int curr_x = mv.X - 4;
                int curr_y = mv.Y - 4;
                int next_x = curr_x + mv.Dx;
                int next_y = curr_y + mv.Dy;
                next_x = max(0, min(next_x, curRawImageGray.cols));
                next_y = max(0, min(next_y, curRawImageGray.rows));
                for(int j = 0; j < 8; ++j)
                {
                    for(int i = 0; i < 8; ++i)
                    {
                        residualFrame.at<float>(curr_y+j, curr_x+i)
                                = float(curRawImageGray.at<u_int8_t>(next_y+j, next_x+i) - preRawImageGray.at<u_int8_t>(curr_y+j, curr_x+i));
                    }
                }
            }
            else if(mv.SegmCode == '-')
            {
                int curr_x = mv.X - 8;
                int curr_y = mv.Y - 4;
                int next_x = curr_x + mv.Dx;
                int next_y = curr_y + mv.Dy;
                next_x = max(0, min(next_x, curRawImageGray.cols));
                next_y = max(0, min(next_y, curRawImageGray.rows));
                for(int j = 0; j < 8; ++j)
                {
                    for(int i = 0; i < 16; ++i)
                    {
                        residualFrame.at<float>(curr_y+j, curr_x+i)
                                = float(curRawImageGray.at<u_int8_t>(next_y+j, next_x+i) - preRawImageGray.at<u_int8_t>(curr_y+j, curr_x+i));
                    }
                }
            }
            else if(mv.SegmCode == '|')
            {
                int curr_x = mv.X - 4;
                int curr_y = mv.Y - 8;
                int next_x = curr_x + mv.Dx;
                int next_y = curr_y + mv.Dy;
                next_x = max(0, min(next_x, curRawImageGray.cols));
                next_y = max(0, min(next_y, curRawImageGray.rows));
                for(int j = 0; j < 16; ++j)
                {
                    for(int i = 0; i < 8; ++i)
                    {
                        residualFrame.at<float>(curr_y+j, curr_x+i)
                                = float(curRawImageGray.at<u_int8_t>(next_y+j, next_x+i) - preRawImageGray.at<u_int8_t>(curr_y+j, curr_x+i));
                    }
                }
            }
            else if(mv.SegmCode == ' ')
            {
                int curr_x = mv.X - 8;
                int curr_y = mv.Y - 8;
                int next_x = curr_x + mv.Dx;
                int next_y = curr_y + mv.Dy;
                next_x = max(0, min(next_x, curRawImageGray.cols));
                next_y = max(0, min(next_y, curRawImageGray.rows));
                for(int j = 0; j < 16; ++j)
                {
                    for(int i = 0; i < 16; ++i)
                    {
                        residualFrame.at<float>(curr_y+j, curr_x+i)
                                = float(curRawImageGray.at<u_int8_t>(next_y+j, next_x+i) - preRawImageGray.at<u_int8_t>(curr_y+j, curr_x+i));
                    }
                }
            }
            else if(mv.SegmCode == '?')
            {
            }
            else
            {
                throw std::runtime_error("Block SegmCode Error!\n");
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
                        residual.at<float>(blk_j*dctGridStep+j, blk_i*dctGridStep+i) = dct_block.at<float>(j, i);
            }
        }

//        subtract(curRawImageGray, preRawImageGray, residual);

        frame.rsd = residual.clone();

        preFrame = curFrame;
        preRawImageGray = curRawImageGray;
    }

    Mat Quantize(const Mat& src)
    {
        Mat dst(src.rows, src.cols, CV_32FC1);
        for(int j = 0; j < src.rows; ++j)
        {
            for(int i = 0; i < src.cols; ++i)
            {
                dst.at<float>(j, i) = round(src.at<float>(j, i));
            }
        }
        return dst;
    }
};

#endif // RESIDUAL_H
