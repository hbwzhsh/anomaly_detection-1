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

        int mb_width = curRawImageGray.cols/gridStep;
        int mb_height = curRawImageGray.rows/gridStep;

        if(firstFlag == true)
        {
            preFrame = curFrame = frame;
            preRawImageGray = curRawImageGray;
            firstFlag = false;
            frame.rsd = Mat::ones(mb_height, mb_width, CV_8UC1);
            return;
        }

        preFrame = curFrame;
        curFrame = frame;
        preRawImageGray = curRawImageGray;

        residual = Mat::zeros(mb_height, mb_width, CV_MAKETYPE(curRawImageGray.depth(), curRawImageGray.channels()));
        residualFrame = Mat::zeros(preRawImageGray.rows, preRawImageGray.cols, CV_MAKETYPE(curRawImageGray.depth(), curRawImageGray.channels()));

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
                for(int j = 0; j < 8; ++j)
                {
                    for(int i = 0; i < 8; ++i)
                    {
                        residualFrame.at<int8_t>(curr_y+j, curr_x+i)
                                = curRawImageGray.at<int8_t>(next_y+j, next_x+i)
                                - preRawImageGray.at<int8_t>(curr_y+j, curr_x+i);
                    }
                }
            }
            else if(mv.SegmCode == '-')
            {
                int curr_x = mv.X - 8;
                int curr_y = mv.Y - 4;
                int next_x = curr_x + mv.Dx;
                int next_y = curr_y + mv.Dy;
                for(int j = 0; j < 8; ++j)
                {
                    for(int i = 0; i < 16; ++i)
                    {
                        residualFrame.at<int8_t>(curr_y+j, curr_x+i)
                                = curRawImageGray.at<int8_t>(next_y+j, next_x+i)
                                - preRawImageGray.at<int8_t>(curr_y+j, curr_x+i);
                    }
                }
            }
            else if(mv.SegmCode == '|')
            {
                int curr_x = mv.X - 4;
                int curr_y = mv.Y - 8;
                int next_x = curr_x + mv.Dx;
                int next_y = curr_y + mv.Dy;
                for(int j = 0; j < 16; ++j)
                {
                    for(int i = 0; i < 8; ++i)
                    {
                        residualFrame.at<int8_t>(curr_y+j, curr_x+i)
                                = curRawImageGray.at<int8_t>(next_y+j, next_x+i)
                                - preRawImageGray.at<int8_t>(curr_y+j, curr_x+i);
                    }
                }
            }
            else if(mv.SegmCode == ' ')
            {
                int curr_x = mv.X - 8;
                int curr_y = mv.Y - 8;
                int next_x = curr_x + mv.Dx;
                int next_y = curr_y + mv.Dy;
                for(int j = 0; j < 16; ++j)
                {
                    for(int i = 0; i < 16; ++i)
                    {
                        residualFrame.at<int8_t>(curr_y+j, curr_x+i)
                                = curRawImageGray.at<int8_t>(next_y+j, next_x+i)
                                - preRawImageGray.at<int8_t>(curr_y+j, curr_x+i);
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

        for(int mb_j = 0; mb_j < mb_height; ++mb_j)
        {
            for(int mb_i = 0; mb_i < mb_width; ++mb_i)
            {
                int sum = 0;
                for(int j = 0; j < gridStep; ++j)
                {
                    for(int i = 0; i < gridStep; ++i)
                    {
//                        sum += residualFrame.at<int8_t>(mb_j*gridStep+j, mb_i*gridStep+i);
                        sum += pow(float(residualFrame.at<int8_t>(mb_j*gridStep+j, mb_i*gridStep+i)), 2.0);
                    }
                }
//                residual.at<int8_t>(mb_j, mb_i) = sum/(gridStep*gridStep);
                residual.at<int8_t>(mb_j, mb_i) = (int8_t)sqrt(sum);
            }
        }

        frame.rsd = residual.clone();
    }

//    void Update(Frame& frame)
//    {
//        if(frame.RawImage.empty())
//        {
//            return;
//        }

//        if(frame.RawImage.channels() != 1)
//        {
//            cvtColor(frame.RawImage, curRawImageGray, CV_BGR2GRAY);
//        }
//        else
//        {
//            curRawImageGray = frame.RawImage;
//        }

//        if(firstFlag == true)
//        {
//            preFrame = curFrame = frame;
//            preRawImageGray = curRawImageGray;
//            firstFlag = false;
//            frame.rsd = Mat::ones(frame.Dx.rows, frame.Dx.cols, CV_8UC1);
//            return;
//        }

//        preFrame = curFrame;
//        curFrame = frame;
//        preRawImageGray = curRawImageGray;

//        residual = Mat::zeros(preFrame.Dx.rows, preFrame.Dx.cols, CV_MAKETYPE(curRawImageGray.depth(), curRawImageGray.channels()));
//        residualFrame = Mat::zeros(preRawImageGray.rows, preRawImageGray.cols, CV_MAKETYPE(curRawImageGray.depth(), curRawImageGray.channels()));

//        for(int blk_j = 0; blk_j < preFrame.Dx.rows; ++blk_j)
//        {
//            for(int blk_i = 0; blk_i < preFrame.Dy.cols; ++blk_i)
//            {
//                int next_blk_j = (blk_j*gridStep + gridStep/2 + (preFrame.Dy)(blk_j, blk_i))/gridStep;
//                int next_blk_i = (blk_i*gridStep + gridStep/2 + (preFrame.Dx)(blk_j, blk_i))/gridStep;
//                next_blk_j = max(0, min(next_blk_j, preFrame.Dx.rows));
//                next_blk_i = max(0, min(next_blk_i, preFrame.Dx.cols));

//                int sum = 0;
//                for(int j = 0; j < gridStep; ++j)
//                {
//                    for(int i = 0; i < gridStep; ++i)
//                    {
////                        residualFrame.at<int8_t>(blk_j*gridStep+j, blk_i*gridStep+i)
////                                = curRawImageGray.at<int8_t>(next_blk_j*gridStep+j, next_blk_i*gridStep+i)
////                                    - preRawImageGray.at<int8_t>(blk_j*gridStep+j, blk_i*gridStep+i);
//                        sum += (curRawImageGray.at<int8_t>(next_blk_j*gridStep+j, next_blk_i*gridStep+i)
//                                - preRawImageGray.at<int8_t>(blk_j*gridStep+j, blk_i*gridStep+i));
//                    }
//                }
////                for(int j = 0; j < gridStep; ++j)
////                {
////                    for(int i = 0; i < gridStep; ++i)
////                    {
////                        residualFrame.at<int8_t>(blk_j*gridStep+j, blk_i*gridStep+i) = sum/(gridStep*gridStep);
////                    }
////                }
//                residual.at<int8_t>(blk_j, blk_i) = sum/(gridStep*gridStep);
//            }
//        }

//        frame.rsd = residual.clone();
//    }
};

#endif // RESIDUAL_H
