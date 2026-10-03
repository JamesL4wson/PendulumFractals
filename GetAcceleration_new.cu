
#define DESIRED_ERROR

__device__
double3 GetAcceleration(const double3& theta, const double3& omega)
{   
    // ------- compute RHS -------

    const double sinXY = sin(theta.x - theta.y);
    const double sinXZ = sin(theta.x - theta.z);
    const double sinYZ = sin(theta.y - theta.z);

    //compute C * omega
    const double row1 = 2 * omega.y * omega.y * sinXY + omega.y * omega.y * sinXZ;
    const double row2 = 2 * omega.x * omega.x * -sinXY + omega.z * omega.z * sinYZ;
    const double row3 = omega.x * omega.x * -sinXZ + omega.y * omega.y * -sinYZ;

    //subtract G
    double rhs[3] = {
        -row1 - 3 * sin(theta.x), 
        -row2 - 2 * sin(theta.y),
        -row3 -     sin(theta.z)
    };

    // ------- M inverse -------

    const double A = cos(theta.x - theta.y);
    const double B = cos(theta.x - theta.z);
    const double C = cos(theta.y - theta.z);
    
    const double invDet = 1.0 / (6.0 - A*A - 2.0*B*B - 3.0*C*C + 2.0*A*B*C);

    const double invM[3][3];
    
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

    const double3 acceleration;

    acceletation.x = rhs[0] * invM[0][0] + rhs[1] * invM[0][1] + rhs[1] * invM[0][2];
    acceletation.y = rhs[0] * invM[1][0] + rhs[1] * invM[1][1] + rhs[1] * invM[1][2];
    acceletation.z = rhs[0] * invM[2][0] + rhs[1] * invM[2][1] + rhs[1] * invM[2][2];

    return acceleration;
}

__device__
State AdaptiveRK(const State& state, double& dt)
{
    while (True) 
    {
        State RK4adjustedState = RK4step(state, dt);
        State RK5adjustedState = RK5step(state, dt);
    
        double error = norm(RK4adjustedState, RK5adjustedState);
    
        if (error > DESIRED_ERROR) {
            dt *= pow(DESIRED_ERROR / error, 0.2);
            continue;
        }
        
        return RK5adjustedState;
    }
}

__device__
State RK4step(const State& state, double dt)
{
    const State k1 = ODE( state );
    const State k2 = ODE( state_add_scaled(state, k1, dt * 0.5) );
    const State k3 = ODE( state_add_scaled(state, k2, dt * 0.5) );
    const State k4 = ODE( state_add_scaled(state, k3, dt) );

    double3 avgThetaVec = add3(k1.theta, add3(scale3(k2.theta, 2.0), add3(scale3(k3.theta, 2.0), k4.theta)));
    double3 avgOmegaVec = add3(k1.omega, add3(scale3(k2.omega, 2.0), add3(scale3(k3.omega, 2.0), k4.omega)));

    State newState;

    const double scale = dt / 6.0;
    newState.theta = add3(state.theta, scale3(avgThetaVec, scale));
    newState.omega = add3(state.omega, scale3(avgOmegaVec, scale));

    return newState;
}

