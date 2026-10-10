#include "Output.cuh"
#include "Parameters.cuh"

constexpr std::size_t OUTPUT_WIDTH = 58;

constexpr double timePerStep = 0.00000000252;
constexpr double predictedRuntime = (timePerStep 
    * N * N
    * WIDTH * HEIGHT 
    * TOTAL_TIME_SECONDS / TIME_STEP_SECONDS
);

std::chrono::system_clock::time_point startStored;

#define PBSTR "############################################################"

void PrintField(const std::string& label, const double value)
{
    std::cout << std::left << std::setfill('.') << std::setw(OUTPUT_WIDTH/2) 
        << label 
        << std::right << std::setfill('.') << std::setw(OUTPUT_WIDTH/2) 
        << value << std::endl;
}

void PrintSolverInfo(
    std::chrono::system_clock::time_point& start
)
{
    std::time_t legacy_time = std::chrono::system_clock::to_time_t(start);
    std::tm* local_tm = std::localtime(&legacy_time);

    startStored = start;

    std::cout << std::string(OUTPUT_WIDTH, '-') << '\n';
    std::cout << "======================= Parameters =======================\n";
    std::cout << std::string(OUTPUT_WIDTH, '-') << '\n';
    std::cout << "\n";

    PrintField("N", N);
    // printField("Masses",                  formatFloats(MASSES));
    // printField("Lengths",                 formatFloats(LENGTHS));
    // printField("Initial Velocities",      formatFloats(INITIAL_VELOCITIES));

    std::cout << "\n";

    PrintField("Horizontal Resolution", WIDTH);
    PrintField("Vertical Resolution", HEIGHT);
    PrintField("Frames", FRAMES);

    std::cout << "\n";

    PrintField("RK4 Step-size (s)", TIME_STEP_SECONDS);
    PrintField("Iteration Time (s)", TOTAL_TIME_SECONDS);

    std::cout << "\n";

    PrintField("Lyapunov Precision", LYAPUNOV_PREC);

    std::cout << "\n";

    std::cout << std::string(OUTPUT_WIDTH, '-') << '\n';
    std::cout << "==================== Running Solver ======================\n";
    std::cout << std::string(OUTPUT_WIDTH, '-') << '\n';
    std::cout << '\n';

    std::cout << "Outputting to filename: " << "TestSolver2.png" << "\n\n";

    std::cout << "Execution Began: "
              << std::put_time(local_tm, "%Y-%m-%d %H:%M:%S") << "\n"; 

    std::cout << "Estimated Required Runtime (s): "
              << predictedRuntime << "\n\n";
}

void PrintCompletedInfo(
    std::chrono::system_clock::time_point& start,
    std::chrono::system_clock::time_point& end
)
{
    auto uptime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::time_t legacy_time = std::chrono::system_clock::to_time_t(end);
    std::tm* local_tm = std::localtime(&legacy_time);

    std::cout << "\n\n";

    std::cout << std::string(OUTPUT_WIDTH, '-') << '\n';
    std::cout << "======================= Finished =========================\n";
    std::cout << std::string(OUTPUT_WIDTH, '-') << '\n';
    std::cout << "\n";

    std::cout << "Execution Ended: "
              << std::put_time(local_tm, "%Y-%m-%d %H:%M:%S") << "\n"; 
    
    std::cout << "Total Uptime (s): "
              << uptime.count() / 1000 << "\n\n";
}

void PrintProgress() {
    auto now = std::chrono::system_clock::now();
    auto nowSeconds =  std::chrono::duration_cast<std::chrono::seconds>(now - startStored);
    double progress = nowSeconds.count() / predictedRuntime;

    int percentage = progress * 100;
    int portionOfBar = progress * (OUTPUT_WIDTH - 10);
    int remainingBar = (OUTPUT_WIDTH - 10) - portionOfBar;

    printf("\r%3d%% [%.*s%*s]", percentage, portionOfBar, PBSTR, remainingBar, "");
    fflush(stdout);
}