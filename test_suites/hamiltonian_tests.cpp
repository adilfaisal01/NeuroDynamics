#include <eigen3/Eigen/Dense>
#include <cmath>
#include <math.h>
#include <iostream>

struct DoublePendulum{

    // take the python parameter
    double m1,m2,l1,l2;
    const double g{9.81};

    struct Result {
      double H;
      double deltaH_max;
    };

    Eigen::Vector4d dynamics(const Eigen::Vector4d& x) const
    {
        double theta1=x(0), omega1=x(1), theta2=x(2),omega2=x(3);
        double diff_theta=theta1-theta2;
        double denom1= l1*(2*m1+m2-(m2*cos(2*diff_theta)));
        double denom2= l2*(2*m1+m2-(m2*cos(2*diff_theta)));
        double num1=-g*(2*m1+m2)*sin(theta1)-m2*g*sin(theta1-2*theta2)-(2*sin(diff_theta)*m2*(pow(omega2,2)*l2+(pow(omega1,2)*l1*cos(diff_theta))));
        double num2=2*sin(diff_theta)*(pow(omega1,2)*l1*(m1+m2)+g*(m1+m2)*cos(theta1)+pow(omega2,2)*l2*m2*cos(diff_theta));

        double alpha1=num1/denom1;
        double alpha2=num2/denom2;
        Eigen::Vector4d xdot;
        xdot << omega1, alpha1, omega2,alpha2;
        return xdot;   
    }

    double Hamiltonian(const Eigen::Vector4d& x) const
    {
        double theta1=x(0), omega1=x(1), theta2=x(2),omega2=x(3);
        double V= -(m1+m2)*g*l1*cos(theta1)-m2*g*l2*cos(theta2);
        double t1= 0.5*(m1*pow(l1,2)*pow(omega1,2));
        double t2= 0.5*m2*(pow(l1,2)*pow(omega1,2)+pow(l2,2)*pow(omega2,2)+2*l1*l2*omega1*omega2*cos(theta1-theta2));
        double T= t1+t2;
        double H=T+V;
        return H;
    }

    Result rungekuttamethod(double T, double dt, double theta1_0, double theta2_0, double omega1_0, double omega2_0) const
    {
        Eigen::Vector4d x;
        x<< theta1_0, omega1_0, theta2_0, omega2_0;
        double H0= Hamiltonian(x); //initial hamiltonian
        double max_drift=0.0;
        for (double t{0}; t<=T; t+=dt) 
        {
            Eigen::Vector4d k1=dynamics(x);
            Eigen::Vector4d k2=dynamics(x+(k1*0.5*dt));
            Eigen::Vector4d k3=dynamics(x+(k2*0.5*dt));
            Eigen::Vector4d k4=dynamics(x+(k3*dt));

            x+=dt/6*(k1+2*k2+2*k3+k4);

            double deltaH=(Hamiltonian(x)-H0)/H0;
            if (std::abs(deltaH)>=max_drift) 
            {
                max_drift=std::abs(deltaH);
            }
        }
        return {H0,max_drift};
    }
};

int main()
{
    DoublePendulum dp{1.0, 0.5, 1.0, 0.75};  // m1, m2, l1, l2
    auto result = dp.rungekuttamethod(10.0, 0.001,  // T=10s, dt=0.001
                                 M_PI/3, 0.0,  // θ₁₀, ω₁₀
                                 M_PI/4, 0.0); // θ₂₀, ω₂₀
    
    if (result.deltaH_max < 1e-6) {
        std::cout << "ayy"<<"\n";
    }

    std::cout << result.deltaH_max<<"\n";
    return 0;
}