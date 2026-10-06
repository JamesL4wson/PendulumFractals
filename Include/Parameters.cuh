#pragma once

#define GRAVITATIONAL_ACCELERATION 9.81

// ==============================================

#define N 3 //the number of pendulums

#define MASSES 1, 1, 1
#define LENGTHS 1, 1, 1

#define INITIAL_VELOCITIES 0, 0, 0

// ==============================================

#define TIME_STEP_SECONDS 0.01
#define TOTAL_TIME_SECONDS 30

#define LYAPUNOV_PREC 0.001 //difference of compared initial states

// ==============================================

#define WIDTH 1000 //horizontal resolution 
#define HEIGHT 1000 //vertical resolution

#define FRAMES 1 //number of images to generate
