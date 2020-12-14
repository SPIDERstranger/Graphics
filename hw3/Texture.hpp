//
// Created by LEI XU on 4/27/19.
//

#ifndef RASTERIZER_TEXTURE_H
#define RASTERIZER_TEXTURE_H
#include "global.hpp"
#include <eigen3/Eigen/Eigen>
#include <opencv2/opencv.hpp>
class Texture{
private:
    cv::Mat image_data;

public:
    Texture(const std::string& name)
    {
        image_data = cv::imread(name);
        cv::cvtColor(image_data, image_data, cv::COLOR_RGB2BGR);
        width = image_data.cols;
        height = image_data.rows;
    }

    int width, height;

    Eigen::Vector3f getColor(float u, float v)
    {
        auto u_img = u * width;
        auto v_img = (1 - v) * height;
        auto color = image_data.at<cv::Vec3b>(v_img, u_img);
        return Eigen::Vector3f(color[0], color[1], color[2]);
    }

    Eigen::Vector3f getColorBilinear(float u, float v)
    {
        u *= width;
        v  = (1-v)*height;
        float minU,maxU,minV,maxV;
        minU = std::floor(u);
        minV = std::floor(v);
        maxU = std::ceil(u);
        maxV = std::ceil(v);
        auto up =  image_data.at<cv::Vec3b>(maxV,minU)*((maxU-u)) + image_data.at<cv::Vec3b>(maxV,maxU)*((u-minU));
        auto down  =  image_data.at<cv::Vec3b>(minV,minU)*((maxU-u)) + image_data.at<cv::Vec3b>(minV,maxU)*((u-minU));
        
        auto center = down *(maxV-v)+up*(v-minV);

        return Eigen::Vector3f(center[0], center[1], center[2]);
    }
};
#endif //RASTERIZER_TEXTURE_H
