#include "ODE.cuh"
#include "Parameters.cuh"

__device__
constexpr double masses[]  = {MASSES};
static_assert(sizeof(masses) / sizeof(masses[0]) == N);
__device__
constexpr double lengths[] = {LENGTHS};
static_assert(sizeof(lengths) / sizeof(lengths[0]) == N);

__device__
constexpr Vect DefaultGravityVec()
{
    Vect g{};

    for (int i = 0; i < N; ++i) {
        double sum = 0.0;
        for (int k = i; k < N; k++) {
            sum += masses[k];
        }
        g[i] = sum * GRAVITATIONAL_ACCELERATION;
    }

    return g;
}
__device__
constexpr Matr DefaultMassMat()
{
    Matr M{};

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            double sum = 0.0;
            for (int k = max(i, j); k < N; k++) {
                sum += masses[k];
            }
            M[i][j] = sum * lengths[j];
        }
    }

    return M;
}
__device__
constexpr Matr DefaultCoreolisMat()
{
    Matr C = DefaultMassMat();
    return C;
}

__device__
Vect GetAccelerationGeneral(const Vect& theta, const Vect& omega)
{
    // =============== Create the rhs vector ===============

    Vect rhs = DefaultGravityVec();
    for (int i = 0; i < N; i++) {
        rhs[i] *= sin(theta[i]);
    }

    // subtract the product C * Omega from the rhs.
    // By noting the symetry of the C matrix, the product can be computed in-place. 
    for (int i = 0; i < N; i++) {
        for (int j = i+1; j < N; j++) {
            double sinDelta = sin(theta[i] - theta[j]);
            double c = DefaultCoreolisMat()[i][j];

            rhs[i] -= c * sinDelta * omega[j] * omega[j];
            rhs[j] += c * sinDelta * omega[i] * omega[i];
        }
    }

    // ======= Create and decompose the mass matrix ======== 

    Matr M = DefaultMassMat();
    for (int i = 0; i < N; i++) {
        for (int j = i+1; j < N; j++) {
            M[i][j] *= cos(theta[i] - theta[j]);
            M[j][i] = M[i][j];
        }
    }

    Vect D; // D is diagonal matrix, only diagonals stored
    Matr L;
    LDLTDecomp(M, D, L);

    // ============== Solve for acceleration ==============

    // Lz = rhs
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < i-1; k++) {
            rhs[i] -= L[i][k] * rhs[k];
        }
    }

    // Dy = rhs
    for (int i = 0; i < N; i++){
        rhs[i] /= D[i];
    }

    //  L^Tx = rhs
    for (int i = N-1; i >= 0; i--) {
        for (int k = i+1; k < N; k++) {
            rhs[i] -= L[i][k] * rhs[k]; //swap i <-> k 
        }
    }

    return rhs;
}

__device__
void LDLTDecomp(const Matr& M, const Vect& D, const Matr& L) 
{
    Cholesky(M, L);

    for (int i = 0; i < N; i++) {
		D[i] = L[i][i] * L[i][i];

        const double inv = 1.0 / L[i][i];

        for (int j = i; j < N; j++) {
            L[j][i] *= inv;
        }
    }
}

__device__
void Cholesky(const Matr& M, const Matr& L)
{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < i; j++) {
            double sum = 0.0;
            for (int k = 0; k < j; k++) {
                sum += M[i][k] * M[j][k];
            }

            L[i][j] = (M[i][j] - sum) / M[j][j];
        }

        // i == j case
        double sum = 0.0;
        for (int k = 0; k < i; k++) {
            sum += M[i][k] * M[i][k];
        }
        L[i][i] = sqrt(M[i][i] - sum);
    }
}

// ====================================================================

__device__
Vect GetAcceleration_special3(const Vect& theta, const Vect& omega)
{   
    // ------- compute RHS -------

    const double sinXY = sin(theta[0] - theta[1]);
    const double sinXZ = sin(theta[0] - theta[2]);
    const double sinYZ = sin(theta[1] - theta[2]);

    //compute C * omega
    const double row1 = 2 * omega[1] * omega[1] * sinXY + omega[2] * omega[2] * sinXZ;
    const double row2 = 2 * omega[0] * omega[0] * -sinXY + omega[2] * omega[2] * sinYZ;
    const double row3 = omega[0] * omega[0] * -sinXZ + omega[1] * omega[1] * -sinYZ;

    //subtract G
    double rhs[3] = {
        -row1 - 3 * sin(theta[0]) * GRAVITATIONAL_ACCELERATION, 
        -row2 - 2 * sin(theta[1]) * GRAVITATIONAL_ACCELERATION,
        -row3 -     sin(theta[2]) * GRAVITATIONAL_ACCELERATION
    };

    // ------- M inverse -------

    const double A = 2.0 * cos(theta[0] - theta[1]);
    const double B = cos(theta[0] - theta[2]);
    const double C = cos(theta[1] - theta[2]);
    
    const double invDet = 1.0 / (6.0 - A*A - 2.0*B*B - 3.0*C*C + 2.0*A*B*C);

    double invM[3][3];
    
    // Diagonal terms
    invM[0][0] = (2.0 - C * C) * invDet;
    invM[1][1] = (3.0 - B * B) * invDet;
    invM[2][2] = (6.0 - A * A) * invDet;

    // Uppoer-diagonal terms
    invM[0][1] = (B * C - A) * invDet;
    invM[0][2] = (A * C - 2.0 * B) * invDet;
    invM[1][2] = (A * B - 3.0 * C) * invDet;

    // Lower-diagonal terms (symetrical)
    invM[1][0] = invM[0][1];
    invM[2][0] = invM[0][2];
    invM[2][1] = invM[1][2];

    // ------- M^-1 * RHS -------

    Vect acceleration;

    acceleration[0] = rhs[0] * invM[0][0] + rhs[1] * invM[1][0] + rhs[2] * invM[2][0];
    acceleration[1] = rhs[0] * invM[0][1] + rhs[1] * invM[1][1] + rhs[2] * invM[2][1];
    acceleration[2] = rhs[0] * invM[0][2] + rhs[1] * invM[1][2] + rhs[2] * invM[2][2];

    return acceleration;
}

// ====================================================================

__device__
State ODE(const State& state)
{
    State step;

    step.theta = state.omega;
    step.omega = GetAccelerationGeneral(state.theta, state.omega);

    return step;
}
