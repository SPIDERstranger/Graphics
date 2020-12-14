#include <iostream>
#include<eigen3/Eigen/Core>

using namespace std;

int main(){
    int a = 0;
    cin>>a;
    while(a>=0)
    {
        cout<<a<<endl;
        cin>>a;
    }
    Eigen::Vector4f b;
    b<<1,1,1,1;
    b.template head<3>();


}