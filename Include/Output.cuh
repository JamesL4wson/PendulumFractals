#include <sstream>
#include <iostream>
#include <iomanip>
#include <chrono>

void PrintField(const std::string& label, const double value);
void PrintSolverInfo(std::chrono::system_clock::time_point& start);
void PrintCompletedInfo(
    std::chrono::system_clock::time_point& start,
    std::chrono::system_clock::time_point& end
);
void PrintProgress();