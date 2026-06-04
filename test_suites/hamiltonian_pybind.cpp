#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <eigen3/Eigen/Dense>
#include <cmath>

namespace py = pybind11;

struct DoublePendulum {
    double m1, m2, l1, l2;
    const double g{9.81};

    struct Result {
        double H0;
        double deltaH_max;
    };

    Eigen::Vector4d dynamics(const Eigen::Vector4d& x) const {
        double theta1 = x(0), omega1 = x(1), theta2 = x(2), omega2 = x(3);
        double diff_theta = theta1 - theta2;
        double denom1 = l1 * (2 * m1 + m2 - (m2 * cos(2 * diff_theta)));
        double denom2 = l2 * (2 * m1 + m2 - (m2 * cos(2 * diff_theta)));
        double num1 = -g * (2 * m1 + m2) * sin(theta1)
                      - m2 * g * sin(theta1 - 2 * theta2)
                      - (2 * sin(diff_theta) * m2 * (pow(omega2, 2) * l2 + (pow(omega1, 2) * l1 * cos(diff_theta))));
        double num2 = 2 * sin(diff_theta) * (pow(omega1, 2) * l1 * (m1 + m2)
                      + g * (m1 + m2) * cos(theta1) + pow(omega2, 2) * l2 * m2 * cos(diff_theta));

        double alpha1 = num1 / denom1;
        double alpha2 = num2 / denom2;
        Eigen::Vector4d xdot;
        xdot << omega1, alpha1, omega2, alpha2;
        return xdot;
    }

    double Hamiltonian(const Eigen::Vector4d& x) const {
        double theta1 = x(0), omega1 = x(1), theta2 = x(2), omega2 = x(3);
        double V = -(m1 + m2) * g * l1 * cos(theta1) - m2 * g * l2 * cos(theta2);
        double t1 = 0.5 * (m1 * pow(l1, 2) * pow(omega1, 2));
        double t2 = 0.5 * m2 * (pow(l1, 2) * pow(omega1, 2)
                    + pow(l2, 2) * pow(omega2, 2)
                    + 2 * l1 * l2 * omega1 * omega2 * cos(theta1 - theta2));
        double T = t1 + t2;
        return T + V;
    }

    Result rungekuttamethod(double T, double dt,
                            double theta1_0, double theta2_0,
                            double omega1_0, double omega2_0) const {
        Eigen::Vector4d x;
        x << theta1_0, omega1_0, theta2_0, omega2_0;
        double H0 = Hamiltonian(x);
        double max_drift = 0.0;
        for (double t = 0; t <= T; t += dt) {
            Eigen::Vector4d k1 = dynamics(x);
            Eigen::Vector4d k2 = dynamics(x + (k1 * 0.5 * dt));
            Eigen::Vector4d k3 = dynamics(x + (k2 * 0.5 * dt));
            Eigen::Vector4d k4 = dynamics(x + (k3 * dt));
            x += dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4);

            double deltaH = (Hamiltonian(x) - H0) / H0;
            if (std::abs(deltaH) >= max_drift) {
                max_drift = std::abs(deltaH);
            }
        }
        return {H0, max_drift};
    }

    /// Convenience: run simulation and return max relative energy drift.
    /// Takes batch state (theta1, theta2, omega1, omega2) as a list of 4 floats.
    /// Returns (H0, deltaH_max).
    py::tuple verify(double T, double dt,
                     double theta1_0, double theta2_0,
                     double omega1_0, double omega2_0) const {
        auto result = rungekuttamethod(T, dt, theta1_0, theta2_0, omega1_0, omega2_0);
        return py::make_tuple(result.H0, result.deltaH_max);
    }
};

PYBIND11_MODULE(_hamiltonian, m) {
    m.doc() = "C++ Hamiltonian verifier for double pendulum parameter ID";

    py::class_<DoublePendulum>(m, "DoublePendulum")
        .def(py::init<double, double, double, double>(),
             py::arg("m1"), py::arg("m2"), py::arg("l1"), py::arg("l2"))
        .def("verify", &DoublePendulum::verify,
             py::arg("T"), py::arg("dt"),
             py::arg("theta1_0"), py::arg("theta2_0"),
             py::arg("omega1_0"), py::arg("omega2_0"),
             "Simulate double pendulum and return (H0, deltaH_max)")
        .def("hamiltonian", [](const DoublePendulum& self,
                                double theta1, double omega1,
                                double theta2, double omega2) {
             Eigen::Vector4d x;
             x << theta1, omega1, theta2, omega2;
             return self.Hamiltonian(x);
        }, "Compute Hamiltonian at a given state")
        .def_readonly("m1", &DoublePendulum::m1)
        .def_readonly("m2", &DoublePendulum::m2)
        .def_readonly("l1", &DoublePendulum::l1)
        .def_readonly("l2", &DoublePendulum::l2);
}