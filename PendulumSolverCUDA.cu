#include "Parameters.cuh"
#include "PendulumSolverCUDA.cuh"
#include "DiffEqSolverCUDA.cuh"

#define PI 3.1415926535897932384626433832795

__device__
constexpr State LyapunovOffset() {
    State lyanpunovOffset;
    for (int i = 0; i < N; i++) {
        lyapunovOffset.theta[i] = LYAPUNOV_PREC;
        lyapunovOffset.omega[i] = 0;
    }
    return lyapunovOffset;
}

__device__
double GetLyapExp(const State& sEnd, const State& sEndNeighbour)
{
    const double delStart = sqrt(3 * LYAPUNOV_PREC * LYAPUNOV_PREC);
    const double delEnd = ErrorBetweenStates(sEnd, sEndNeighbour);

    return 1.0/TOTAL_TIME_SECONDS * log(delEnd / delStart);
}

__global__
void SolvePendulum_kernal(double* divergences)
{
    constexpr int totalCases = WIDTH * HEIGHT;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= totalCases) {return;}

    Vect angles = GetAngleSpacePos(index);

    const State initialState = {angles, {INITIAL_VELOCITIES}};
    const State initialStateNeighbour = initialState + LyapunovOffset();

    State currentState = initialState;
    State stateNeighbour = initialStateNeighbour;

    constexpr int totalSteps = static_cast<int>(TOTAL_TIME_SECONDS / TIME_STEP_SECONDS);
    for (int i = 1; i < totalSteps; i++)
    {
        currentState = RK4Step(currentState, TIME_STEP_SECONDS);
        stateNeighbour = RK4Step(stateNeighbour, TIME_STEP_SECONDS);
    }

    divergences[index] = GetLyapExp(currentState, stateNeighbour);
}

__device__
void GetAngleSpacePos(const int index) 
{
    constexpr double dx = 2.0 * PI / static_cast<double>(WIDTH);
    constexpr double dy = 2.0 * PI / static_cast<double>(HEIGHT);

    const int x = index / HEIGHT;
    const int y = index % HEIGHT;

    Vect angles = {
        x * dx - PI,
        y * dy - PI,
        PI / 2
    };

    return Vect;
}