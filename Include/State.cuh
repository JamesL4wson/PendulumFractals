#pragma once

#include <cuda_runtime.h>
#include <cuda/std/array>

#include "Parameters.cuh"

typedef cuda::std::array<double, N> Vect;
typedef cuda::std::array<cuda::std::array<double, N>, N> Matr;

__host__ __device__
inline Vect add(const Vect& a, const Vect& b)
{
    Vect result{};
    for (int i = 0; i < N; ++i)
        result[i] = a[i] + b[i];
    return result;
}
__host__ __device__
inline Vect sub(const Vect& a, const Vect& b)
{
    Vect result{};
    for (int i = 0; i < N; ++i)
        result[i] = a[i] - b[i];
    return result;
}
__host__ __device__
inline Vect scale(const Vect& a, double s)
{
    Vect result{};
    for (int i = 0; i < N; ++i)
        result[i] = a[i] * s;
    return result;
}
__host__ __device__
inline double dot(const Vect& a, const Vect& b)
{
    double result = 0.0;
    for (int i = 0; i < N; ++i)
        result += a[i] * b[i];
    return result;
}

struct State
{
    Vect theta;
    Vect omega;

    __host__ __device__
    inline State operator+(const State& s) const
    {
        return {add(theta, s.theta), add(omega, s.omega)};
    }
    __host__ __device__
    inline State operator-(const State& s) const
    {
        return {sub(theta, s.theta), sub(omega, s.omega)};
    }
    __host__ __device__
    inline State operator*(double s) const
    {
        return {scale(theta, s), scale(omega, s)};
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
    return sqrt(dot(diff.theta, diff.theta) + dot(diff.omega, diff.omega));
}

__device__
inline State axpy(const State& rhs, const State& lhs, double s)
{
    State result{};
    for (int i = 0; i < N; i++)
    {
        result.theta[i] = rhs.theta[i] + s * lhs.theta[i];
        result.omega[i] = rhs.omega[i] + s * lhs.omega[i];
    }
    return result;
}
