#include "DiffEqSolverCUDA.cuh"
#include "ODE.cuh"

#define TOLERANCE 0.001

__device__
void RK4Step(State& state, double dt)
{
    const State k1 = ODE( state );
    const State k2 = ODE( axpy(state, k1, dt * 0.5) );
    const State k3 = ODE( axpy(state, k2, dt * 0.5) );
    const State k4 = ODE( axpy(state, k3, dt) );

    state = axpy(state, (axpy(axpy(k1, k2, 2), k3, 2.0) + k4), dt/6.0);
}

__device__
State AdaptiveRK45(const State& state, double& dt)
{
    while (true) 
    {
        //doubles used in this function are decimal expansions of those
        //chosen for the Runge-Kutta-Fehlberg RKF45 integrator. 
        //As outlined at: http://maths.cnam.fr/IMG/pdf/RungeKuttaFehlbergProof.pdf

        // -------- Sample the ODE in 6 locations --------
        const State k1 = ODE( 
            state 
        );
        const State k2 = ODE( 
            state 
            + 0.25 * k1 
        );
        const State k3 = ODE( 
            state 
            + 0.09375 * k1 
            + 0.28125 * k2
        );
        const State k4 = ODE( 
            state 
            + 0.8793809741 * k1 
            - 3.2771961766 * k2 
            + 3.3208921256 * k3
        );
        const State k5 = ODE( 
            state 
            + 2.03240740741 * k1 
            - 8.0           * k2 
            + 7.1734892788  * k3 
            + 0.2058966862  * k4
        );
        const State k6 = ODE( 
            state 
            - 0.2962962963 * k1 
            + 2.0          * k2 
            - 1.3816764133 * k3 
            + 0.4529727096 * k4  
            - 10.275       * k5
        );

        // -------- Calculate the apropriate RK4/RK5 steps -------
        State rk4 = (
            state 
            + 0.1157407407 * dt * k1
            + 0.5489278752 * dt * k3
            + 0.535331384  * dt * k4
            - 0.3          * dt * k5
        );
        State rk5 = (
            state
            + 0.1185185185  * dt * k1
            + 0.5189863548  * dt * k3
            + 0.5061314903  * dt * k4
            - 0.18          * dt * k5
            + 0.03636363636 * dt * k6
        );

        // -------- Evaluate the error on steps ---------
        // double error = ErrorBetweenStates(rk4, rk5);
        double error = 0;
    
        double stepScale = pow((TOLERANCE * dt) / (error), 0.25);
        stepScale = fminf(fmaxf(stepScale, 0.1), 5.0);

        dt *= stepScale;

        if (error <= TOLERANCE) {
            return rk5;
        }
    }
}
