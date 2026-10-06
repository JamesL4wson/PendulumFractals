#pragma once

#include <cuda_runtime.h>
#include <cuda/std/array>

#include "Parameters.cuh"

typedef cuda::std::array<double, N> Vect;
typedef cuda::std::array<cuda::std::array<double, N>, N> Matr;

__host__ __device__
inline double3 add3(const double3& a, const double3& b)
{
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__
inline double3 sub3(const double3& a, const double3& b)
{
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__
inline double3 scale3(const double3& a, double s)
{
    return make_double3(a.x * s, a.y * s, a.z * s);
}

__host__ __device__
inline double dot3(const double3& a, const double3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

struct State
{
    double3 theta;
    double3 omega;

    __host__ __device__
    inline State operator+(const State& s) const
    {
        return {add3(theta, s.theta), add3(omega, s.omega)};
    }
    __host__ __device__
    inline State operator-(const State& s) const
    {
        return {sub3(theta, s.theta), sub3(omega, s.omega)};
    }
    __host__ __device__
    inline State operator*(double s) const
    {
        return {scale3(theta, s), scale3(omega, s)};
    }
};
__host__ __device__
inline State operator*(double s, const State& x)
{
    return x * s;
}

__device__
inline double ErrorBetweenStates(const State& state1, const State& state2)
{
    const State diff = state1 - state2;
    return sqrt(dot3(diff.theta, diff.theta) + dot3(diff.omega, diff.omega));
}

