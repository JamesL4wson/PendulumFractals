#include "Parameters.cuh"
#include "PendulumSolverCUDA.cuh"
#include "DiffEqSolverCUDA.cuh"

#define PI 3.1415926535897932384626433832795

__device__
double GetLyapExp(const State& sEnd, const State& sEndNeighbour)
{
    constexpr double delStart = sqrt(6 * LYAPUNOV_PREC * LYAPUNOV_PREC);
    const double delEnd = ErrorBetweenStates(sEnd, sEndNeighbour);

    return 1.0/10 * log(delEnd / delStart);
}

__global__
void SolvePendulum_kernal(double* divergences, double angle3)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    constexpr int totalCases = WIDTH * HEIGHT;

    if (index >= totalCases)
        return;
        
    constexpr double dx = 2.0 * PI / static_cast<double>(WIDTH);
    constexpr double dy = 2.0 * PI / static_cast<double>(HEIGHT);

    const int x = index / HEIGHT;
    const int y = index % HEIGHT;

    const double angle1 = x * dx - PI;
    const double angle2 = y * dy - PI;

    const State initialState = {
        make_double3(angle1, angle2, angle3),
        {INITIAL_VELOCITIES}
    };
    const State initialStateNeighbour = {
        make_double3(angle1 + LYAPUNOV_PREC, angle2 + LYAPUNOV_PREC, angle3 + LYAPUNOV_PREC),
        {INITIAL_VELOCITIES}
    };

    State currentState = initialState;
    State stateNeighbour = initialStateNeighbour;

    constexpr int totalSteps = static_cast<int>(TOTAL_TIME_SECONDS / TIME_STEP_SECONDS);
    for (int i = 1; i < totalSteps; ++i)
    {
        currentState = RK4Step(currentState, TIME_STEP_SECONDS);
        stateNeighbour = RK4Step(stateNeighbour, TIME_STEP_SECONDS);
    }

    divergences[index] = GetLyapExp(currentState, stateNeighbour);
}
