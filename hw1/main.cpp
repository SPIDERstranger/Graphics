#include "Triangle.hpp"
#include "rasterizer.hpp"
#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <opencv2/opencv.hpp>

constexpr double MY_PI = 3.1415926;

Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos)
{
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f translate;
    translate << 1, 0, 0, -eye_pos[0],
        0, 1, 0, -eye_pos[1],
        0, 0, 1, -eye_pos[2],
        0, 0, 0, 1;

    view = translate * view;

    return view;
}

Eigen::Matrix4f get_model_matrix(float rotation_angle)
{
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();

    // TODO: Implement this function
    // Create the model matrix for rotating the triangle around the Z axis.
    // Then return it.

    Eigen::Matrix4f translate;
    while (rotation_angle >= 360)
    {
        rotation_angle -= 360;
    }
    while (rotation_angle < 0)
    {
        rotation_angle += 360;
    }
    rotation_angle = rotation_angle / 180.0 * MY_PI;
    translate << cosf(rotation_angle), -sinf(rotation_angle), 0, 0,
        sinf(rotation_angle), cosf(rotation_angle), 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1;

    model = translate * model;

    return model;
}

Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio,
                                      float zNear, float zFar)
{
    // Students will implement this function

    Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();

    // TODO: Implement this function
    // Create the projection matrix for the given parameters.
    // Then return it.
    // 计算正交投影
    Eigen::Matrix4f pto = Eigen::Matrix4f::Zero();
    pto(0, 0) = zNear;
    pto(1, 1) = zNear;
    pto(2, 2) = zFar + zNear;
    pto(3, 2) = 1;
    pto(2, 3) = -zNear * zFar;

    // std::cout<<pto<<std::endl;

    // 通过角度计算矩阵的每个 lr tb nf
    float top = zNear * tanf(MY_PI * eye_fov / 360.0);
    float bottom = -top;
    float right = top * aspect_ratio;
    float left = -right;

    // 计算正交压缩
    Eigen::Matrix4f oScale;
    oScale << 2 / (right - left), 0, 0, 0,
        0, 2 / (top - bottom), 0, 0,
        0, 0, 2 / (zFar - zNear), 0,
        0, 0, 0, 1;

    //计算正交移动
    Eigen::Matrix4f oMove = Eigen::Matrix4f::Identity();
    oMove(0, 3) = -(right + left) / 2.0;
    oMove(1, 3) = -(top + bottom) / 2.0;
    oMove(2, 3) = -(zNear + zFar) / 2.0;

    projection = oScale * oMove * pto * projection;

    return projection;
}


Eigen::Matrix4f get_rotation(Eigen::Vector3f axis,float angle)
{
    Eigen::Matrix3f model = Eigen::Matrix3f::Identity();
    Eigen::Matrix3f n;
    n<<0,-axis.z(),axis.y(),
    axis.z(),0,-axis.z(),
    -axis.y(),axis.z(),0;
    float rotate = MY_PI*angle/180.0;
    model = model*cos(rotate) +(1-cos(rotate))*axis*axis.transpose() + sin(rotate)*n;
    Eigen::Matrix4f result=Eigen::Matrix4f::Identity();
    result.block(0,0,3,3) = model;
    return result;
}
int main(int argc, const char **argv)
{
    float angle = 0;
    bool command_line = false;
    std::string filename = "output.png";

    if (argc >= 3)
    {
        command_line = true;
        angle = std::stof(argv[2]); // -r by default
        if (argc == 4)
        {
            filename = std::string(argv[3]);
        }
        else
            return 0;
    }

    rst::rasterizer r(700, 700);

    Eigen::Vector3f eye_pos = {0, 0, 5};

    std::vector<Eigen::Vector3f> pos{{2, 0, -2}, {0, 2, -2}, {-2, 0, -2}};

    std::vector<Eigen::Vector3i> ind{{0, 1, 2}};

    auto pos_id = r.load_positions(pos);
    auto ind_id = r.load_indices(ind);

    int key = 0;
    int frame_count = 0;

    if (command_line)
    {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, rst::Primitive::Triangle);
        // std::cout << "draw" << std::endl;
        // for (auto a : r.frame_buffer())
        // {
        //     std::cout << a.transpose() << std::endl;
        // }
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);

        cv::imwrite(filename, image);

        return 0;
    }

    while (key != 27)
    {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, rst::Primitive::Triangle);
        // std::cout << "draw" << std::endl;
        // for (auto a : r.frame_buffer())
        // {
        //     std::cout << a.transpose() << std::endl;
        // }

        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::imshow("image", image);
        key = cv::waitKey(10);

        std::cout << "frame count: " << frame_count++ << '\n';

        std::cin >> key;
        if (key == 'a')
        {
            angle += 10;
        }
        else if (key == 'd')
        {
            angle -= 10;
        }
    }

    return 0;
}
