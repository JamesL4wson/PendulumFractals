#include <cuda_runtime.h>

#include "State.cuh"

__device__
State RK4Step(const State& state, double dt);
__device__
State AdaptiveRK45(const State& state, double& dt);