#include <cuda_runtime.h>
#include <cuda/std/array>

#include "State.cuh"

#define GRAVITATIONAL_ACCELERATION 9.81

__device__
Vect GetAccelerationGeneral(const Vect& theta, const Vect& omega);
__device__
void LDLTDecomp(Matr& M, Vect& D);
__device__
void Cholesky_inPlace(Matr& M);
__device__
double3 GetAcceleration_special3(const double3& theta, const double3& omega);
__device__
State ODE(const State& state);
