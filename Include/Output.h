#include <sstream>

void PrintField(const std::string& label, const double value);
void PrintSolverInfo(std::__1::chrono::system_clock::time_point& start);
void PrintCompletedInfo(
    std::__1::chrono::system_clock::time_point& start,
    std::__1::chrono::system_clock::time_point& end
);
void PrintProgress();