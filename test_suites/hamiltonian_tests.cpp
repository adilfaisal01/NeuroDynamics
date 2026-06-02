#include <Eigen/Dense>
#include <cmath>
#include <math.h>

struct DoublePendulum{

    // take the python parameter
    double m1,m2,l1,l2;

    //simulation initial conditions
    double theta1, theta2, omega1, omega2;

    //time setups,
    double T, dt;

    const double g{9.81};

    double diff_theta=theta1-theta2;

    Eigen::Vector4d dynamics()
    {
        double denom1= l1*(2*m1+m2-(m2*cos(2*diff_theta)));
        double denom2= l2*(2*m2+m2-(m2*cos(2*diff_theta)));
        double num1=-g*(2*m1+m2)*sin(theta1)-m2*g*sin(theta1-2*theta2)-(2*sin(diff_theta)*m2*(pow(omega2,2)*l2+(pow(omega1,2)*l1*cos(diff_theta))));
        double num2=2*sin(diff_theta)*(pow(omega1,2)*l1*(m1+m2)+g*(m1+m2)*cos(theta1)+pow(omega2,2)*l2*m2*cos(diff_theta));

        double alpha1=num1/denom1;
        double alpha2=num2/denom2;

        Eigen::Vector4d xdot;
        xdot<< omega1, omega2, alpha1,alpha2;
        return xdot;
        
    }
};