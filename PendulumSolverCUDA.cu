#include "PendulumSolverCUDA.cuh"
#include "DiffEqSolverCUDA.cuh"

__device__
double GetLyapExp(const State& sEnd, const State& sEndNeighbour)
{
    const double delStart = sqrt(6 * LYAP_EPSILON * LYAP_EPSILON);
    const double delEnd   = ErrorBetweenStates(sEnd, sEndNeighbour);

    return 1.0/10 * log(delEnd / delStart);
}

__global__
void SolvePendulum_kernal(double* divergences, double angle3)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    const int totalCases = WIDTH * HEIGHT;

    if (index >= totalCases)
        return;

    const int x = index / HEIGHT;
    const int y = index % HEIGHT;

    const double dx = 2.0 * PI / static_cast<double>(WIDTH);
    const double dy = 2.0 * PI / static_cast<double>(HEIGHT);

    const double angle1 = x * dx - PI;
    const double angle2 = y * dy - PI;

    const State initialState = {
        make_double3(angle1, angle2, angle3),
        make_double3(0.0, 0.0, 0.0)
    };
    const State initialStateNeighbour = {
        make_double3(angle1 + LYAP_EPSILON, angle2 + LYAP_EPSILON, angle3 + LYAP_EPSILON),
        make_double3(0.0, 0.0, 0.0)
    };

    State currentState = initialState;
    State stateNeighbour = initialStateNeighbour;

    const int totalSteps = static_cast<int>(TOTAL_TIME_SECONDS / STEP_SIZE);
    for (int i = 1; i < totalSteps; ++i)
    {
        currentState = RK4Step(currentState, STEP_SIZE);
        stateNeighbour = RK4Step(stateNeighbour, STEP_SIZE);
    }

    divergences[index] = GetLyapExp(currentState, stateNeighbour);
}
