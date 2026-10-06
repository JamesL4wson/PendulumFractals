#include <cuda_runtime.h>
#include <iostream>

#include "State.cuh"
#include "ODE.cuh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__device__
double ErrorBetweenStates(const State& state1, const State& state2);

__device__
double GetLyapExp(const State& sEnd, const State& sEndNeighbour);

__global__
void SolvePendulum_kernal(double* divergences, double angle3);