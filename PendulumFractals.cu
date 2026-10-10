#include <cuda_runtime.h>

#include <pngwriter.h>

#include <cmath>
#include <stdexcept>
#include <vector>
#include <fstream>
#include <chrono>
#include <thread>

#include "PendulumSolverCUDA.cuh"
#include "Output.cuh"

#define PI 3.1415926535897932384626433832795

std::vector<float> CreateColor(double diveregence)
{
    const int col1[3] = {0,0,0};
    const int col2[3] = {1,1,1};

    float R = (col2[0] - col1[0]) * diveregence + col1[0];
    float G = (col2[1] - col1[1]) * diveregence + col1[1];
    float B = (col2[2] - col1[2]) * diveregence + col1[2];

    std::vector<float> returnColor = {R, G, B};

    return returnColor;
}

void CreateImage(std::vector<double>& divergences) 
{
    pngwriter image(WIDTH, HEIGHT, 1.0, "DoublePendulumFractal.png");

    for (int x = 0; x < WIDTH; x++)
    {
        for (int y = 0; y < HEIGHT; y++)
        {
            std::vector<float> color = CreateColor(divergences[x * WIDTH + y]);
            image.plot(x, y, color[0], color[1], color[2]);
        }
    }

    image.close();
}


std::vector<double> ComputeDivergences()
{
    constexpr int totalCases = WIDTH * HEIGHT;
    
    double* divergences = nullptr;
    gpuErrchk( cudaMalloc(&divergences, totalCases * sizeof(double)) );
    
    constexpr int threadsPerBlock = 256;
    const int blocks = (totalCases + threadsPerBlock - 1) / threadsPerBlock;
    SolvePendulum_kernal<<<blocks, threadsPerBlock>>>(divergences);

    cudaEvent_t event;
    gpuErrchk( cudaEventCreate(&event) );
    gpuErrchk( cudaEventRecord(event, 0) );

    while (cudaEventQuery(event) == cudaErrorNotReady) 
    {
        PrintProgress();
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    gpuErrchk( cudaEventDestroy(event) );
    gpuErrchk( cudaGetLastError() );
    
    std::vector<double> result(WIDTH * HEIGHT);
    gpuErrchk( cudaMemcpy(result.data(), divergences, totalCases * sizeof(double), cudaMemcpyDeviceToHost) );

    gpuErrchk( cudaFree(divergences) );

    return result;
}

int main()
{
    auto start = std::chrono::system_clock::now();
    PrintSolverInfo(start);

    std::vector<double> divs = ComputeDivergences();
    CreateImage(divs);

    auto stop = std::chrono::system_clock::now();
    PrintCompletedInfo(start, stop);
}
