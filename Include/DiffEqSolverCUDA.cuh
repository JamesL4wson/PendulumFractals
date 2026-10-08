#include <cuda_runtime.h>

#include "State.cuh"

__device__
void RK4Step(State& state, double dt);
__device__
State AdaptiveRK45(const State& state, double& dt);
