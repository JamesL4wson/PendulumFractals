#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>

#define GRAVITATIONAL_ACCELERATION 9.81

using Vect = Eigen::VectorXd;
using Matr = Eigen::MatrixXd;

std::vector<Vect> EvolvePendulum(const Vect& initalPosition, int time, double dt);
Vect RK4Step(const Vect& state, double dt);
Vect NPendulumODE(const Vect& s);
Vect Acceleration(const Vect& theta, const Vect& omega);
double CalculateDivergence(const Vect& finalPos, const Vect& initalPos);

int main() {
    std::ofstream out("divergences.txt");

    int screenWidth = 100;
    int screenHeight = 100;

    int numberOfPendulums = 3;
    int time = 30;
    double stepSize = 0.01;

    double dx = 2 * M_PI / screenWidth;
    double dy = 2 * M_PI / screenHeight;

    double angle1 = 0; double angle2 = 0;
    for (int x = 0; x < screenWidth; x++, angle1 += dx) 
    {   
        std::cout << x << ", " << angle1 << "\n";

        for (int y = 0; y < screenHeight/2; y++, angle2 += dy)
        {
            Vect initalPosition(numberOfPendulums * 2);
            initalPosition << angle1, angle2, M_PI/2, 0, 0, 0;
            
            std::vector<Vect> solution = EvolvePendulum(initalPosition, time, stepSize);

            double divergence = CalculateDivergence(solution.back(), initalPosition);

            out << divergence << "\n";
        }
        angle2 = 0; 
    }

    out.close();
    return 0;
}

std::vector<Vect> EvolvePendulum(const Vect& initalPosition, int time, double dt) 
{
    int totalSteps = time / dt;

    std::vector<Vect> solution;
    solution.resize(totalSteps);

    solution[0] = initalPosition;
    for (int i = 1; i < totalSteps; i++) {
        solution[i] = RK4Step(solution[i-1], dt);
    }

    return solution;
}

Vect RK4Step(const Vect& state, double dt) 
{
    Vect f1 = NPendulumODE(state);
    Vect f2 = NPendulumODE(state + dt/2.0 * f1);
    Vect f3 = NPendulumODE(state + dt/2.0 * f2);
    Vect f4 = NPendulumODE(state + dt * f3);

    return state + dt/6.0 * (f1 + 2.0*f2 + 2.0*f3 + f4);
}

Vect NPendulumODE(const Vect& s)
{
    int n = s.rows() / 2;

    Vect p = s.head(n);
    Vect v = s.tail(n);
    Vect a = Acceleration(p, v);

    Vect step(2 * n);

    step.head(n) = v;
    step.tail(n) = a;

    return step; 
}

Vect Acceleration(const Vect& p, const Vect& v)
{
    double g = GRAVITATIONAL_ACCELERATION;
    int n = p.rows();

    Matr M = Matr::Zero(n, n);
    Matr C = Matr::Zero(n, n);
    Vect G = Vect::Zero(n);

    for (std::size_t i = 0; i < n; ++i)
    {
        G(i) = (n - i) * g * std::sin(p(i));

        for (std::size_t j = 0; j < n; ++j)
        {
            const std::size_t k0 = std::max(i, j);

            const double massSum_k = n - k0;
            const double dtheta = p(i) - p(j);

            M(i,j) = massSum_k * std::cos(dtheta);
            C(i,j) = massSum_k * std::sin(dtheta) * v(j);
        }
    }

    Vect rhs = -(C * v + G);

    return M.ldlt().solve(rhs);
}

double CalculateDivergence(const Vect& finalPos, const Vect& initalPos)
{
    return (finalPos - initalPos).norm(); 
}