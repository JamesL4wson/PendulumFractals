//AI generated slop version of PendulumSolver.cpp translated to CUDA
#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <fstream>
#include <array>

constexpr double GRAVITATIONAL_ACCELERATION = 9.81;
constexpr double PI = 3.1415926535897932384626433832795;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


struct State
{
    double3 theta;
    double3 omega;
};

__host__ __device__
inline double3 add3(const double3& a, const double3& b)
{
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__
inline double3 sub3(const double3& a, const double3& b)
{
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__
inline double3 scale3(const double3& a, double s)
{
    return make_double3(a.x * s, a.y * s, a.z * s);
}

__host__ __device__
inline double dot3(const double3& a, const double3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__
State state_add_scaled(const State& state, const State& derivative, double scale)
{
    State result;

    result.theta = add3(state.theta, scale3(derivative.theta, scale));
    result.omega = add3(state.omega, scale3(derivative.omega, scale));

    return result;
}



__device__
double3 GetAcceleration(const double3& theta, const double3& omega)
{
    double p[3] = {theta.x, theta.y, theta.z};
    double v[3] = {omega.x, omega.y, omega.z};

    double M[3][3];
    double rhs[3];

    for (int i = 0; i < 3; ++i)
    {
        double coriolis = 0.0;
        double gravity = (3 - i) * GRAVITATIONAL_ACCELERATION * sin(p[i]);

        for (int j = 0; j < 3; ++j)
        {
            const int k = (i > j) ? i : j;
            const double massSum = 3.0 - k;
            const double dtheta = p[i] - p[j];

            M[i][j] = massSum * cos(dtheta);

            coriolis += massSum * sin(dtheta) * v[j] * v[j];
        }

        rhs[i] = -(coriolis + gravity);
    }

    double L[3][3] = {
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0}
    };

    double D[3];
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < i; ++j)
        {
            double sum = M[i][j];

            for (int k = 0; k < j; ++k)
            {
                sum -= L[i][k] * L[j][k] * D[k];
            }

            L[i][j] = sum / D[j];
        }

        double diag = M[i][i];

        for (int k = 0; k < i; ++k)
        {
            diag -= L[i][k] * L[i][k] * D[k];
        }

        D[i] = diag;
    }

    double y[3];
    for (int i = 0; i < 3; ++i)
    {
        double sum = rhs[i];

        for (int k = 0; k < i; ++k)
        {
            sum -= L[i][k] * y[k];
        }

        y[i] = sum;
    }

    double z[3];
    for (int i = 0; i < 3; ++i)
    {
        z[i] = y[i] / D[i];
    }

    double a[3];
    for (int i = 2; i >= 0; --i)
    {
        double sum = z[i];

        for (int k = i + 1; k < 3; ++k)
        {
            sum -= L[k][i] * a[k];
        }

        a[i] = sum;
    }

    return make_double3(a[0], a[1], a[2]);
}

__device__
State ODE(const State& state)
{
    State step;

    step.theta = state.omega;
    step.omega = GetAcceleration(state.theta, state.omega);

    return step;
}

__device__
State RK4Step(const State& state, double dt)
{
    const State k1 = ODE( state );
    const State k2 = ODE( state_add_scaled(state, k1, dt * 0.5) );
    const State k3 = ODE( state_add_scaled(state, k2, dt * 0.5) );
    const State k4 = ODE( state_add_scaled(state, k3, dt) );

    double3 avgThetaVec = add3(k1.theta, add3(scale3(k2.theta, 2.0), add3(scale3(k3.theta, 2.0), k4.theta)));
    double3 avgOmegaVec = add3(k1.omega, add3(scale3(k2.omega, 2.0), add3(scale3(k3.omega, 2.0), k4.omega)));

    State newState;

    const double scale = dt / 6.0;
    newState.theta = add3(state.theta, scale3(avgThetaVec, scale));
    newState.omega = add3(state.omega, scale3(avgOmegaVec, scale));

    return newState;
}

__device__
double GetLyapExp(double3& s0, double3& s0Neighbour, double3& sEnd, double3& sEndNeighbour)
{
    const double3 dTheta0 = sub3(s0Neighbour.theta, s0.theta);
    const double3 dOmega0 = sub3(s0Neighbour.omega, s0.omega);
    const double del0 = sqrt(dot3(dTheta0, dTheta0) + dot3(dOmega0, dOmega0));

    const double3 dThetaEnd = sub3(sEndNeighbour.theta, sEnd.theta);
    const double3 dOmegaEnd = sub3(sEndNeighbour.omega, sEnd.omega);
    const double delEnd = sqrt(dot3(dThetaEnd, dThetaEnd) + dot3(dOmegaEnd, dOmegaEnd));

    return log(delEnd / del0);
}

__global__
void SolvePendulum_kernal(double* divergences, int width, int height, double angle3, int time, double dt)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    const int totalCases = width * height;

    if (index >= totalCases)
        return;

    const int x = index / height;
    const int y = index % height;

    const double dx = 2.0 * PI / static_cast<double>(width);
    const double dy = 2.0 * PI / static_cast<double>(height);

    const double angle1 = x * dx;
    const double angle2 = y * dy;

    const State initialState = {
        make_double3(angle1, angle2, angle3),
        make_double3(0.0, 0.0, 0.0)
    };
    const State initialStateNeighbour = {
        make_double3(angle1 + eps, angle2 + eps, angle3 + eps)
        make_double3(0, 0, 0)
    };

    State currentState = initialState;
    State stateNeighbour = initialStateNeighbour;

    const int totalSteps = static_cast<int>(time / dt);
    for (int i = 1; i < totalSteps; ++i)
    {
        currentState = RK4Step(currentState, dt);
        stateNeighour = RK4Step(stateNeighbour, dt);
    }

    divergences[index] = GetLyapExp(initalState, initalStateNeighbour, currentState, stateNeighbour);
}


std::array<float> CreateColor(double diveregence)
{
    static const int[] col1 = [0,0,0];
    static const int[] col2 = [1,1,1];

    float R = (col2[0] - col1[0]) * diveregence + col1[0];
    float G = (col2[1] - col1[1]) * diveregence + col1[1];
    float B = (col2[2] - col1[1]) * diveregence + col1[2];

    std::array<float, 3> returnColor = [R, G, B];

    return returnColor;
}

std::array<double> ComputeDivergences(int width, int height, double angle3, int time, double dt)
{
    const int totalCases = width * height;
    
    double* divergences = nullptr;
    gpuErrchk( cudaMalloc(&divergences, totalCases * sizeof(double)) );
    
    constexpr int threadsPerBlock = 256;
    const int blocks = (totalCases + threadsPerBlock - 1) / threadsPerBlock;
    SolvePendulum_kernal<<<blocks, threadsPerBlock>>>(divergences, width, height, angle3, time, dt);

    gpuErrchk( cudaGetLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    
    std::array<double, wdith*height> result(totalCases);
    gpuErrchk( cudaMemcpy(result.data(), divergences, totalCases * sizeof(double), cudaMemcpyDeviceToHost) );

    gpuErrchk( cudaFree(divergences) );

    return result;
}

int main()
{
    int width = 1000;
    int height = 1000;

    double dt = 0.01;
    int time = 30;

    double angle3 = PI/2;

    std::array<double, width*height> divs = ComputeDivergences(width, height, anlge3, time, dt);

    pngwriter image(width, height, 1.0, "TriplePendulumFractal.png");

    for (int x = 0; x < width; x++)
    {
        for (int y = 0; y < height; y++)
        {
            std::array<float, 3> color = CreateColor(divs[x * wdith + y]);
            image.plot(x, y, color[0], color[1], color[2]);
        }
    }

    image.close();

    return 0;
}