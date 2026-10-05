#include <cuda_runtime.h>
#include <iostream>

#include "State.cuh"
#include "ODE.cuh"

#define GRAVITATIONAL_ACCELERATION 9.81
#define PI 3.1415926535897932384626433832795

#define WIDTH 1000
#define HEIGHT 1000

#define STEP_SIZE 0.05
#define TOTAL_TIME_SECONDS 30
#define LYAP_EPSILON 0.001

struct State;

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