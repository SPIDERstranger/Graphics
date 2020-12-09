#include<cmath>
#include<eigen3/Eigen/Core>
#include<eigen3/Eigen/Dense>
#include<iostream>
#include "rasterizer.hpp"
#include "Triangle.hpp"
using namespace std;
using namespace Eigen;
constexpr double MY_PI = 3.1415926;

Eigen::Matrix4f get_model_matrix(float rotation_angle)
{
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
    
    // TODO: Implement this function
    // Create the model matrix for rotating the triangle around the Z axis.
    // Then return it.
    
    Eigen::Matrix4f translate;
    while(rotation_angle>=360)
    {
        rotation_angle-=360;
    }
    while(rotation_angle<0)
    {
        rotation_angle+=360;
    }
    rotation_angle = rotation_angle/180.0 * MY_PI;
    translate<<
        cosf(rotation_angle),-sinf(rotation_angle),0,0,
        sinf(rotation_angle),cosf(rotation_angle) ,0,0,
        0                   ,0                    ,1,0,
        0                   ,0                    ,0,1;

        model = translate*model;
    return model;
}

Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio,
                                      float zNear, float zFar)
{

    Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();

    // 计算正交投影
    Eigen::Matrix4f pto = Eigen::Matrix4f::Zero();
    pto(0,0) = zNear;
    pto(1,1) = zNear;
    pto(2,2) = zFar+zNear;
    pto(3,2) = 1;
    pto(2,3) = -zNear*zFar;
    
    // std::cout<<pto<<std::endl;
    
    // 通过角度计算矩阵的每个 lr tb nf
    float top = zNear*tanf(MY_PI*eye_fov/360.0);
    float bottom = -top;
    float right = top*aspect_ratio;
    float left = -right;

    // 计算正交压缩
    Eigen::Matrix4f oScale;
    oScale<<
        2/(right-left),0             ,0             ,0,
        0             ,2/(top-bottom),0             ,0,
        0             ,0             ,2/(zFar-zNear),0,
        0             ,0             ,0             ,1;

    //计算正交移动
    Eigen::Matrix4f oMove= Eigen::Matrix4f::Identity();
    oMove(0,3) = (right+left)/2.0;
    oMove(1,3) = (top+bottom)/2.0;
    oMove(2,3) = (zNear+zFar)/2.0;
    std::cout<< oMove<<std::endl;
    projection= oScale*oMove*pto * projection;

    return projection;
}



static bool insideTriangle(int x, int y, const Vector3f* _v)
{   
    // TODO : Implement this function to check if the point (x, y) is inside the triangle represented by _v[0], _v[1], _v[2]
    Vector3f point(x,y,0);
    float lastz = 0;
    for(int i = 0 ;i<3;++i)
    {
        Vector3f edge (_v[i].x()-_v[(i+1)%3].x(),
                                _v[i].y()-_v[(i+1)%3].y(),
                                0);
        Vector3f pe (_v[i].x()-point.x(),
                            _v[i].y()-point.y(),
                            0);
        Vector3f result = edge.cross(pe);

        if(lastz == 0 || (lastz>0&& result.z()>=0)||(lastz<0&&result.z()<=0))
        {
            lastz = result.z();
        }
        else{
            return false;
        }
    }
    return true;
}



int main(){

    std::cout<<get_model_matrix(30)<<std::endl;

    std::cout<<get_projection_matrix(45, 1, 0.1, 50)<<std::endl;
    // // Basic Example of cpp
    // std::cout << "Example of cpp \n";
    // float a = 1.0, b = 2.0;
    // std::cout << a << std::endl;
    // std::cout << a/b << std::endl;
    // std::cout << std::sqrt(b) << std::endl;
    // std::cout << std::acos(-1) << std::endl;
    // std::cout << std::sin(30.0/180.0*acos(-1)) << std::endl;

    // // Example of vector
    // std::cout << "Example of vector \n";
    // // vector definition
    // Eigen::Vector3f v(1.0f,2.0f,3.0f);
    // Eigen::Vector3f w(1.0f,0.0f,0.0f);
    // // // // vector output
    // // // std::cout << "Example of output \n";
    // std::cout << v << std::endl;
    // // // // vector add
    // // // std::cout << "Example of add \n";
    // // // std::cout << v + w << std::endl;
    // // // // vector scalar multiply
    // std::cout << "Example of scalar multiply \n";
    // // std::cout << v * 3.0f << std::endl;
    // std::cout << 2.0f * v << std::endl;
    // std::cout<< v.dot(w) <<std::endl;
    // std::cout<< v.cross(w) <<std::endl;


    std::cout << "Test: insideTriangle  \n";
    
    Vector3f list[3];
    list[0] =  Vector3f(0,0,0);
    list[1] =  Vector3f(3,1,0);
    list[2] =  Vector3f(1,3,0);
    Vector2f check1 (2,2);
    Vector2f check2 (3,2);
    
    cout<<((insideTriangle(check1.x(),check1.y(),list)?" true ":" false "))<<endl;

    check1[0] = 1;
    cout<<check1<<endl;
    cout<<((insideTriangle(check1.x(),check1.y(),list)?" true ":" false "))<<endl;

    cout<<((insideTriangle(check2.x(),check2.y(),list)?" true ":" false "))<<endl;
    
    std::cout << "End Test: insideTriangle  \n";

    

    // Example of matrix
    // std::cout << "Example of matrix \n";
    // // matrix definition
    // Eigen::Matrix3f i,j;
    // i << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0;
    // j << 2.0, 3.0, 1.0, 4.0, 6.0, 5.0, 9.0, 7.0, 8.0;
    // Eigen::Matrix4f test = Eigen::Matrix4f::Identity();
    // std::cout<<test<<std::endl;

    // // matrix output
    // std::cout << "Example of output \n";
    // std::cout << i << std::endl;
    // // matrix add i + j
    // // matrix scalar multiply i * 2.0
    // // matrix multiply i * j
    getchar();
    system("pause");
    return 0;
}