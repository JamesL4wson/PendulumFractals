#include <cuda_runtime.h>

#include <pngwriter.h>

#include <cmath>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>

#include "PendulumSolverCUDA.cuh"

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

void CreateImage(std::vector<double> divergences) 
{
    pngwriter image(WIDTH, HEIGHT, 1.0, "TestSolver2.png");

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
    const int totalCases = WIDTH * HEIGHT;
    
    double* divergences = nullptr;
    gpuErrchk( cudaMalloc(&divergences, totalCases * sizeof(double)) );
    
    constexpr int threadsPerBlock = 256;
    const int blocks = (totalCases + threadsPerBlock - 1) / threadsPerBlock;
    SolvePendulum_kernal<<<blocks, threadsPerBlock>>>(divergences);

    gpuErrchk( cudaGetLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    
    std::vector<double> result(WIDTH * HEIGHT);
    gpuErrchk( cudaMemcpy(result.data(), divergences, totalCases * sizeof(double), cudaMemcpyDeviceToHost) );

    gpuErrchk( cudaFree(divergences) );

    return result;
}

int main()
{
    std::cout << "Execution started...\n";
    auto start =std::chrono::high_resolution_clock::now();

    std::vector<double> divs = ComputeDivergences();
    CreateImage(divs);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Execution finished\n" << "Time taken:" << duration.count() << "\n";
}






constexpr std::size_t OUTPUT_WIDTH = 100;
constexpr std::size_t RULE_WIDTH   = 58;

std::string formatFloat(double value)
{
    std::ostringstream out;
    out << std::fixed << std::setprecision(6) << value;
    return out.str();
}

template <typename Range>
std::string formatFloats(const Range& values)
{
    std::ostringstream out;
    out << '[';

    bool first = true;
    for (const auto& value : values)
    {
        if (!first)
            out << ", ";

        out << std::fixed << std::setprecision(6) << value;
        first = false;
    }

    out << ']';
    return out.str();
}

void printField(const std::string& label, const std::string& value)
{
    const std::size_t used = label.size() + 1 + value.size();

    const std::size_t dotCount =
        (used < OUTPUT_WIDTH) ? OUTPUT_WIDTH - used : 1;

    std::cout << label << ' '
              << std::string(dotCount, '.')
              << value << '\n';
}

void printRule()
{
    std::cout << std::string(RULE_WIDTH, '-') << '\n';
}

void PrintSolverOutput()
{
    constexpr double timePerStep = 0.000000007555555556;
    constexpr double predictedRuntime = (timePerStep 
        * N 
        * WIDTH * HEIGHT 
        * TIME_STEP_SECONDS * TIME_STEP_SECONDS
    );

    constexpr double currentTime = 0;
    constexpr double startTime = 0; 

    printRule();
    std::cout << "=============== Running Solver ================\n";
    printRule();

    std::cout << "Outputting to filename: " << "TestSolver2.png" << "\n\n";

    std::cout << "Execution Began: "
              << formatFloat(startTime) << '\n';

    std::cout << "Up Time: "
              << formatFloat(currentTime - startTime) << "\n\n";

    std::cout << "Estimated Required Runtime: "
              << formatFloat(predictedRuntime) << '\n';

    std::cout << "Estimated Remaining Runtime: "
              << formatFloat(predictedRuntime - currentTime) << "\n\n";

    printRule();
    std::cout << "================= Parameters =================\n";
    printRule();
    std::cout << '\n';

    printField("N",                       formatFloat(N));
    printField("Masses",                  formatFloats(MASSES));
    printField("Lengths",                 formatFloats(LENGTHS));
    printField("Initial Velocities",      formatFloats(INITIAL_VELOCITIES));

    std::cout << '\n';

    printField("Horizontal Resolution",   formatFloat(WIDTH));
    printField("Vertical Resolution",     formatFloat(HEIGHT));
    printField("Frames",                  formatFloat(FRAMES));

    std::cout << '\n';

    printField("RK4 Step-size (s)",       formatFloat(STEP_SIZE_SECONDS));
    printField("Iteration Time (s)",      formatFloat(TOTAL_TIME_SECONDS));

    std::cout << '\n';

    printField("Lyapunov Precision",      formatFloat(LYAPUNOV_PREC));
}
