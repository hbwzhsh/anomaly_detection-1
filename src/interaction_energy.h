#ifndef INTERACTION_ENERGY_H
#define INTERACTION_ENERGY_H

#include "common.h"
#include "log.h"
#include "frame_reader.h"
#include <cstdlib>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.h>

struct InteractEnergy
{
    static const int gridStep = 16;
    static const float alpha = 0.5;
    static const float beta = 0.5;
    Mat_<float> energyMap;
//    bool signedGradient; // false: 180 degree; true: 360 degree
    float distanceThreshold;
    float angleThreshold;
    float radius;

    InteractEnergy(/*bool signedGradient = false,*/ float distanceThreshold = 100.0, float angleThreshold = 30.0, float radius = 3.0)
        :
          /*signedGradient(signedGradient),*/ distanceThreshold(distanceThreshold), angleThreshold(angleThreshold), radius(radius)
    {
    }

    float CalculateDistance(int posX1, int posY1, float dX1, float dY1,
                           int posX2, int posY2, float dX2, float dY2)
    {
        float dis = 0;
        dis = alpha * (pow(float(posX1 - posX2), 2.0) + pow(float(posY1 - posY2), 2.0))
                +
              beta * (pow(dX1 - dX2, 2.0) + pow(dY1 - dY2, 2.0));
        return dis;
    }

    float CalculateAngle(float dX1, float dY1,
                         float dX2, float dY2)
    {
//        float fullAngle = signedGradient ? 360 : 180;
        float angle = fabs(double(fastAtan2(dY2, dX2) - fastAtan2(dY1, dX1)));
        if(angle > 180)
        {
            angle = 360 - angle;
        }
        return angle;
    }

    float CalculateEnergy(int posX1, int posY1, float dX1, float dY1,
                          int posX2, int posY2, float dX2, float dY2)
    {
        float distance = CalculateDistance(posX1, posY1, dX1, dY1, posX2, posY2, dX2, dY2);
        float angle = CalculateAngle(dX1, dY1, dX2, dY2);
        printf("%f, %f\n", distance, angle);
        int w_d;
        int w_a;

        if(distance < distanceThreshold)
            w_d = 1;
        else
            w_d = 0;

        if(angle < angleThreshold)
            w_a = 1;
        else
            w_a = 0;

        float energy = w_d * w_a * exp(-(distance/(2*pow(radius, 2.0))));

        return energy;
    }

    void Update(Frame& frame)
    {
        Size sz = frame.Dx.size();
        energyMap = Mat_<float>::zeros(sz.height, sz.width);

        for(int h = 0; h < sz.height; ++h)
        {
            for(int w = 0; w < sz.width; ++w)
            {
                float interactionEnergy = 0;

                for(int j = 0; j < sz.height; ++j)
                {
                    for(int i = 0; i < sz.width; ++i)
                    {
                        if(h==j && w==i)
                            continue;

                        interactionEnergy += CalculateEnergy(w, h, frame.Dx(h, w), frame.Dy(h, w),
                                                             i, j, frame.Dx(j, i), frame.Dy(j, i));
                    }
                }

                interactionEnergy /= sz.height*sz.width-1;
                energyMap(h, w) = interactionEnergy;
            }
        }

        frame.energyMap = energyMap.clone();
    }
};

#endif // INTERACTION_ENERGY_H
