// pendulum.cu

//AI generated slop version of PendulumSolver.cpp translated to CUDA

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <vector>

constexpr double GRAVITATIONAL_ACCELERATION = 9.81;
constexpr double PI = 3.1415926535897932384626433832795;

struct State
{
    double3 theta;
    double3 omega;
};

__host__ __device__
inline double3 add3(const double3& a, const double3& b)
{
    return make_double3(
        a.x + b.x,
        a.y + b.y,
        a.z + b.z
    );
}

__host__ __device__
inline double3 sub3(const double3& a, const double3& b)
{
    return make_double3(
        a.x - b.x,
        a.y - b.y,
        a.z - b.z
    );
}

__host__ __device__
inline double3 scale3(const double3& a, double s)
{
    return make_double3(
        a.x * s,
        a.y * s,
        a.z * s
    );
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
double3 acceleration(const double3& theta, const double3& omega)
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
State pendulum_ode(const State& s)
{
    State ds;

    ds.theta = s.omega;
    ds.omega = acceleration(s.theta, s.omega);

    return ds;
}

__device__
State rk4_step(const State& state, double dt)
{
    const State k1 = pendulum_ode(state);

    const State k2 =
        pendulum_ode(
            state_add_scaled(
                state,
                k1,
                dt * 0.5
            )
        );

    const State k3 =
        pendulum_ode(
            state_add_scaled(
                state,
                k2,
                dt * 0.5
            )
        );

    const State k4 =
        pendulum_ode(
            state_add_scaled(
                state,
                k3,
                dt
            )
        );

    double3 theta_sum =
        add3(
            k1.theta,
            add3(
                scale3(k2.theta, 2.0),
                add3(
                    scale3(k3.theta, 2.0),
                    k4.theta
                )
            )
        );

    double3 omega_sum =
        add3(
            k1.omega,
            add3(
                scale3(k2.omega, 2.0),
                add3(
                    scale3(k3.omega, 2.0),
                    k4.omega
                )
            )
        );

    State result;

    const double scale = dt / 6.0;
    result.theta = add3(state.theta, scale3(theta_sum, scale));
    result.omega = add3(state.omega, scale3(omega_sum, scale));

    return result;
}

__global__
void evolve_pendulums(double* divergences, int screenWidth, int screenHeight, int time, double dt)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    const int rows = screenHeight / 2;
    const int totalCases = screenWidth * rows;

    if (index >= totalCases)
        return;

    const int x = index / rows;
    const int y = index % rows;

    const double dx = 2.0 * PI / static_cast<double>(screenWidth);
    const double dy = 2.0 * PI / static_cast<double>(screenHeight);

    const double angle1 = x * dx;
    const double angle2 = y * dy;

    const State initialState = {
        make_double3(angle1, angle2, PI / 2.0),
        make_double3(0.0, 0.0,0.0)
    };

    State state = initialState;

    const int totalSteps = static_cast<int>(time / dt);
    for (int i = 1; i < totalSteps; ++i)
    {
        state = rk4_step(state, dt);
    }

    const double3 dtheta = sub3(state.theta, initialState.theta);
    const double3 domega = sub3(state.omega, initialState.omega);
    const double divergence = sqrt(dot3(dtheta, dtheta) + dot3(domega, domega));

    divergences[index] = divergence;
}

#define CUDA_CHECK(call)                                             \
    do                                                               \
    {                                                                \
        cudaError_t err = (call);                                    \
        if (err != cudaSuccess)                                      \
        {                                                            \
            throw std::runtime_error(                               \
                std::string("CUDA error: ") +                        \
                cudaGetErrorString(err)                             \
            );                                                       \
        }                                                            \
    } while (0)

std::vector<double> compute_divergences(int width, int height, int time, double dt)
{
    const int totalCases = width * height;

    double* d_divergences = nullptr;
    CUDA_CHECK(cudaMalloc(&d_divergences, totalCases * sizeof(double)));
    
    constexpr int threadsPerBlock = 256;
    const int blocks = (totalCases + threadsPerBlock - 1) / threadsPerBlock;
    evolve_pendulums<<<blocks, threadsPerBlock>>>(d_divergences, width, height, time, dt);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    std::vector<double> result(totalCases);
    CUDA_CHECK(cudaMemcpy(result.data(), d_divergences, totalCases * sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_divergences));

    return result;
}