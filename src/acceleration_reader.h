#ifndef ACCELERATION_READER_H
#define ACCELERATION_READER_H

#include "common.h"
#include <vector>
#include <deque>

using namespace std;
using namespace cv;

struct AccelerationReader
{
    static const int gridStep = 16;
    int iInterval;
    int iFrameNum;
    deque< Mat_<float> > MvStackX;
    deque< Mat_<float> > MvStackY;
    Mat_<float> Ax;
    Mat_<float> Ay;

    AccelerationReader()
        :
          iInterval(1),
          iFrameNum(iInterval+2)
    {
    }

    AccelerationReader(int Interval)
        :
          iInterval(Interval),
          iFrameNum(iInterval+2)
    {
    }

    bool Update(Frame& f)
    {
        if(MvStackX.size() >= iFrameNum)
        {
            MvStackX.pop_front();
            MvStackY.pop_front();
        }

        MvStackX.push_back(f.Dx);
        MvStackY.push_back(f.Dy);

        if(MvStackX.size() == iFrameNum)
        {
            ComputeAcceleration();
            return true;
        }

        return false;
    }

    void ComputeAcceleration()
    {
        Size sz = MvStackX.front().size();
        Ax = Mat_<float>::zeros(sz);
        Ay = Mat_<float>::zeros(sz);

        Mat_<float> backDx = MvStackX.back();
        Mat_<float> frontDx = MvStackX.front();
        Mat_<float> backDy = MvStackY.back();
        Mat_<float> frontDy = MvStackY.front();

        for(int j = 0; j < sz.height; ++j)
        {
            for(int i = 0; i < sz.width; ++i)
            {
                int next_i = (i*gridStep + gridStep/2 + frontDx(j, i))/gridStep;
                int next_j = (j*gridStep + gridStep/2 + frontDy(j, i))/gridStep;

                next_i = max(0, min(next_i, sz.width-1));
                next_j = max(0, min(next_j, sz.height-1));

                Ax(j, i) = backDx(next_j, next_i) - frontDx(j, i);
                Ay(j, i) = backDy(next_j, next_i) - frontDy(j, i);
            }
        }
    }

//    void WriteData(Mat& matData)
//    {
//        if(matData.empty())
//        {
//            cout<<"Mat empty!"<<endl;
//            return;
//        }

//        for(int r=0; r<matData.rows; ++r)
//        {
//            for(int c=0; c<matData.cols; ++c)
//            {
//                cout<<matData.at<float>(r,c)<<'\t';
//            }
//            cout<<endl;
//        }
//    }
};

#endif // ACCELERATION_READER_H
