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
        curFrame = frame;

        int mb_width = curRawImageGray.cols/gridStep;
        int mb_height = curRawImageGray.rows/gridStep;

        if(firstFlag == true)
        {
            preFrame = curFrame = frame;
            preRawImageGray = curRawImageGray;
            firstFlag = false;
            frame.rsd = Mat::zeros(mb_height, mb_width, CV_32FC1);
            return;
        }

        residual = Mat::zeros(mb_height, mb_width, CV_32FC1);
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

        for(int mb_j = 0; mb_j < mb_height; ++mb_j)
        {
            for(int mb_i = 0; mb_i < mb_width; ++mb_i)
            {
                float sum = 0;
                for(int j = 0; j < gridStep; ++j)
                {
                    for(int i = 0; i < gridStep; ++i)
                    {
                        sum += residualFrame.at<float>(mb_j*gridStep+j, mb_i*gridStep+i);
//                        sum += pow(float(residualFrame.at<int8_t>(mb_j*gridStep+j, mb_i*gridStep+i)), 2.0);
                    }
                }
                residual.at<float>(mb_j, mb_i) = sum/(gridStep*gridStep);
//                residual.at<int8_t>(mb_j, mb_i) = (int8_t)sqrt(sum);
            }
        }

        frame.rsd = residual.clone();

        preFrame = curFrame;
        preRawImageGray = curRawImageGray;
    }
};

#endif // RESIDUAL_H
