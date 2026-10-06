# Generate Pendulum Fractals

Pendulums are some of the simplest dynamical systems that one can imagine. Despite this, compound pendulums (a linkage of more than one pendulum) are highly complex and chaotic, with no general analytic solution to their governing differential equations. 

It is however, (perhaps suprisingly) not the case that these systems are always chaotic. For certain angles, there exist periodic solutions to the compound pendulum DEs. The locations in angle space where such solutions occur define highly complex fractals. These fractals are explored for double pemndulums here: https://www.youtube.com/watch?v=dtjb2OhEQcU

This repo can construct these fractals for n-compound pendulums.

## Usage

To use the generator open the Parameters.cpp file and set the various calculator parameters, particuarly N the number of pendulums, but also the masses, lengths, gravity, image resolution, ODE solver accuracy and more. These constants must be set at compile-time. 

Next, build the executable using cmake and an appopriate CUDA compiler. 

Now run the compiled .exe. The generator will take hours for accurate/high-resolution images. 

## Implementaion

The generator is written in CUDA C++ and implements the core differential equation solver on GPU. 

For each pixel in the desired image, the generator will solve the n-compound-pendulum ODE for two similar inital states (differing by some configurable epsilon). It will solve these using the Runge-Kutta-4 algorithm. 

After the desired number of steps, the Lyapunov exponent of these states will be calculated. 

The solver will then plot a colour linearly interpolated between (0, 0, 0) and (255, 255, 255) using this exponent. 
