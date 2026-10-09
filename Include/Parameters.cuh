#pragma once

#define GRAVITATIONAL_ACCELERATION 9.81

// ==============================================

#define N 4 //the number of pendulums

#define MASSES 1.0, 1.0, 1.0, 1.0
#define LENGTHS 1.0, 1.0, 1.0, 1.0

#define INITIAL_VELOCITIES 0, 0, 0, 0

// ==============================================

#define TIME_STEP_SECONDS 0.01
#define TOTAL_TIME_SECONDS 30

#define LYAPUNOV_PREC 0.001 //difference of compared initial states

// ==============================================

#define WIDTH 10000 //horizontal resolution 
#define HEIGHT 10000 //vertical resolution

#define FRAMES 1 //number of images to generate

// ==============================================

#define MIN_ANGLE -PI/2
#define MAX_ANGLE PI/2
