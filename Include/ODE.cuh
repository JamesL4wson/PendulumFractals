#include <cuda_runtime.h>
#include <cuda/std/array>

#include "State.cuh"

// Implements a specialized version of the LDLT decomposition to solve for the acceleration vector
// of an N compound pendulum. Writen for CUDA, but also more or less valid C++ code.
//
// Systems as here: https://link.springer.com/article/10.1007/s11071-025-11862-1
// LDL^T method here: https://nm.mathforcollege.com/mws/gen/04sle/mws_gen_sle_txt_cholesky.pdfs
__device__
Vect GetAccelerationGeneral(const Vect& theta, const Vect& omega);
__device__
void LDLTDecomp(Matr& M, Vect& D);
__device__
void Cholesky_inPlace(Matr& M);
// Specialises the pendulum ODE to 3 pendulums
__device__
double3 GetAcceleration_special3(const double3& theta, const double3& omega);
__device__
State ODE(const State& state);
