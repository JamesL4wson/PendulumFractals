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
    LDLTDecomp(M, D);   // decomposes M in-place, M is now its lower diagonal factor L

    // ============== Solve for acceleration ==============

    // Lz = rhs
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < i; k++) {
            rhs[i] -= M[i][k] * rhs[k];
        }
    }

    // Dy = rhs
    for (int i = 0; i < N; i++){
        rhs[i] /= D[i];
    }

    //  L^Tx = rhs
    for (int i = N-1; i >= 0; i--) {
        for (int k = i+1; k < N; k++) {
            rhs[i] -= M[k][i] * rhs[k];
        }
    }

    return rhs;
}

__device__
void LDLTDecomp(Matr& M, Vect& D) 
{
    Cholesky_inPlace(M); // reusing M for efficiency, M <-> L

    for (int i = 0; i < N; i++) {
		D[i] = M[i][i] * M[i][i];

        const double inv = 1.0 / M[i][i];

        for (int j = i; j < N; j++) {
            M[j][i] *= inv;
        }
    }
}
__device__
void Cholesky_inPlace(Matr& M)
{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < j; k++) {
                sum += M[i][k] * M[j][k];
            }
            
            M[i][j] = (M[i][j] - sum) / M[j][j];
        }
    }
}

// ====================================================================

__device__
double3 GetAcceleration_special3(const double3& theta, const double3& omega)
{   
    // ------- compute RHS -------

    const double sinXY = sin(theta.x - theta.y);
    const double sinXZ = sin(theta.x - theta.z);
    const double sinYZ = sin(theta.y - theta.z);

    //compute C * omega
    const double row1 = 2 * omega.y * omega.y * sinXY + omega.z * omega.z * sinXZ;
    const double row2 = 2 * omega.x * omega.x * -sinXY + omega.z * omega.z * sinYZ;
    const double row3 = omega.x * omega.x * -sinXZ + omega.y * omega.y * -sinYZ;

    //subtract G
    double rhs[3] = {
        -row1 - 3 * sin(theta.x) * GRAVITATIONAL_ACCELERATION, 
        -row2 - 2 * sin(theta.y) * GRAVITATIONAL_ACCELERATION,
        -row3 -     sin(theta.z) * GRAVITATIONAL_ACCELERATION
    };

    // ------- M inverse -------

    const double A = 2.0 * cos(theta.x - theta.y);
    const double B = cos(theta.x - theta.z);
    const double C = cos(theta.y - theta.z);
    
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

    double3 acceleration;

    acceleration.x = rhs[0] * invM[0][0] + rhs[1] * invM[1][0] + rhs[2] * invM[2][0];
    acceleration.y = rhs[0] * invM[0][1] + rhs[1] * invM[1][1] + rhs[2] * invM[2][1];
    acceleration.z = rhs[0] * invM[0][2] + rhs[1] * invM[1][2] + rhs[2] * invM[2][2];

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
