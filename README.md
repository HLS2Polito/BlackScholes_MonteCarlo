Monte Carlo Methods applied to the Black-Scholes financial market model
============
### Table of contents
[toc]


## Overview
### Black-Scholes Model
The [Black-Scholes model][Black-Scholes Model], which was first published by Fischer Black and Myron Scholes in 1973, is a famous and basic mathematical model describing the behaviour of investment instruments in financial markets. This model focuses on comparing the Return On Investment for one risky asset, whose price is subject to [geometric Brownian motion][geometric Brownian motion] and one riskless asset with a fixed interest rate. 

The geometric Brownian behaviour of the price of the risky asset is described by this stochastic differential equation:

$$dS=rSdt+\sigma SdW_t$$

where $S$ is the price of the risky asset (usually called stock price), $r$ is the fixed interest rate of the riskless asset, $\sigma$ is the volatility of the stock and $W_t$ is a [Wiener process][Wiener process].

According to Itoh'sī Lemma, the analytical solution of this stochastical  differential equation is as follows:

$$ S_{t+\Delta t}=e^{(r-\frac{1}{2}\sigma^2)\Delta t+\sigma\epsilon\sqrt{\Delta t} } $$

where $\epsilon\sim N(0,1)$  (the standard normal distribution).

### Call/Put Option
Entering more specifically into the financial sector operations, two styles of stock transaction [options][option] are considered in this project, namely the European vanilla option and Asian option (which is one of the [exotic options][exotic options]). 
[Call options][Call options] and [put options][put options] are defined reciprocally. Given the basic parameters for an option, namely expiration date and strike price, the call/put payoff price could be estimated as follows. 
For the European vanilla option, we have:

$$P_{Call}=max\{S-K,0\}\\P_{put}=max\{K-S,0\}$$

where $S$ is the stock price at the expiration date (estimated by the model above) and $K$ is the strike price.
For the Asian option, we have:

$$P_{Call}=max\{\frac{1}{T}\int_0^TSdt-K,0\}\\P_{put}=max\{K-\frac{1}{T}\int_0^TSdt,0\}$$

where $T$ is the time period (between now and the option expiration date) , $S$ is the stock price at the expiration date, and $K$ is the strike price.

[option]: https://en.wikipedia.org/wiki/Option_style
[exotic option]: https://en.wikipedia.org/wiki/Exotic_option

### The Monte Carlo Method
The [Monte Carlo Method][Monte Carlo] is one of the most widely used approaches to simulate stochastic processes, like a stock price modeled with Black-Scholes. This is especially true for exotic options, which are usually not solvable analytically. In this project, the Monte Carlo Method is used to estimate the payoff price of a given instrument using the Black Scholes model. 

The given time period has to be partitioned into M steps according to the style of the option. M=1 for a European option, since the payoff price is independent of the price before the expiration date. At each time point of a Monte Carlo simulation of this kind, the stock price is determined by the stock price at the previous time point and by a normally distributed random number. The expectation of the payoff price can thus be estimated by N parallel independent simulations.

The convergence of the result produced by the Monte Carlo method is ensured essetially by running a very large number of simulations. $C=MN$ is usually a very large number, e.g. $10^9$. 

### Normally Distributed Random Number Generation
A key aspect of the quality of the results of the Monte Carlo method is the quality of the random numbers that it uses. The normally distributed random numbers that are used in this project are generated by the Mersenne-Twister algorithm followed by the Box-Muller transformation. 

#### Mersenne-Twister
The [Mersenne Twister][Mersenne Twister] is an algorithm to generate uniformly distributed pseudorandom numbers. Its very long periodicity $2^{19937}-1$ makes it a suitable algorithm for our application, since as discussed above the Monte Carlo method requires millions of random numbers.

#### Box-Muller transform
The [Box Muller transformation][Box Muller transformation] transforms a pair of independent, uniformly distributed random numbers in the interval (0,1) into a pair of independent normally distributed random numbers, which are required for simulating the Black Scholes model.
Given two independent $U_1$,$U_2 \sim U(0,1)$, 

$$Z_1=\sqrt{-2ln(U_1)}cos(2\pi U_2)
\\ Z_2=\sqrt{-2ln(U_1)}sin(2\pi U_2)$$

then $Z_1$,$Z_2\sim N(0,1)$, also independent.


## Getting Started 
The repository contains two directories, "blackEuro" and "blackAsian", implementing the European option and the Asian option respectively. They are written using C++, rather than OpenCL, because tehre is no workgroup-level memory that is worth sharing. All Monte Carlo simulations are independent.
### Files Tree
```
blackScholes_MonteCarlo
â   README.md
â
âââ headers
â   â   defTypes.h
â   â   RNG.h
â   â   RNG.cpp
â   â   blackScholes.h
â   â   stockData.h
â   â   stockData.cpp
â   ââ  ML_cl.h
â
âââ blackEuro
â   â   solution.tcl
â   â   blackEuro.h
â   â   blackEuro.cpp
â   â   testBench.h
â   â   main.cpp
â   ââ  blackScholes.cpp
â
âââ blackAsian
    â   solution.tcl
    â   blackAsian.h
    â   blackAsian.cpp
    â   testBench.h
    â   main.cpp
    ââ  blackScholes.cpp
```
File/Dir name  |Information
-------------- | ---
blackEuro.cpp  | Top CPU function of the European option kernel
blackAsian.cpp | Top CPU function of the Asian option kernel
solution.tcl   | Script to run sdaccel
blackScholes.h | It declares the blackScholes object instantiated in the top functions (same methods for European and Asian option).
blackScholes.cpp | It defines the blackScholes object instantiated in the top functions. Note that the definitions of the object methods are different between the European And Asian options.
stockData.cpp	 | Basic stock datasets. It defines an object instantiated in the top functions
RNG.cpp   | Random Number Generator class. It defines an object instantiated in
the blackSholes objects.
main.cpp | Host code calling the kernels
testBench.h | Input parameters for the kernels
ML_cl.h | CL/cl.hpp for OpenCL 1.2.6

Note that in the repository we had to include the OpenCL header for version 1.2.6, instead of the version 1.1 installed by sdaccel, because the latter causes compile-time errors. SDAccel and Vivado HLS work perfectly well with this header. See figure ![alt text][clerror]

[clerror]: https://github.com/KitAway/BlackScholes_MonteCarlo/blob/master/figures/header_failure.PNG

### Parameters
The values of the parameters for a given stock and option are listed in ***"testBench.h"***. 

Parameter |  information
:-------- | :---
T	       |  time period
rate       |  interest rate of riskless asset
volatility |  volatility of the risky asset (stock)
S0		   |  initial price of the stock
K          |  strike price for the option
kernel_name | the kernel name, to be passed to the OpenCL runtime

The number of simulations $N$, and the number of time partitions $M$, as well as all the other parameters related to the simulation, are listed in ***"blackScholes.cpp"***

Parameter |  information
:-------- | :---
MAX_NUM_RNG | number of RNGs running in parallel
MAX_SAMPLE  | number of simulations of each group
TIME_PAR    |  $M$
where $N=512*MAX_NUM_RNG*MAX_SAMPLE $ and each parallel simulation group assigned to a given RNG runs 512 simulations.

### How to run an example
In each sub-directory, there is a script file called "solution.tcl". It can be used as follows:
> sdaccel solution.tcl
The result of the call/put payoff price estimation will be printed on standard IO.

Due to some bugs in SDAccel, the kernel (written in C++) cannot be emulated on the CPU right now (see the figure below). Only RTL simulation is available. However, note that RTL simulation takes a very long time. In order to obtain (imprecise) results quickly, the computation cost $C$ can be reduced.

![alt text](https://github.com/KitAway/BlackScholes_MonteCarlo/blob/master/figures/CPU_emulation.PNG)

### Sample Output
For the European option:

Input parameter |  value
:-------- | :---
T| 1
S0  | 100
K 	| 110
rate    |  5%
volatility | 20%
MAX_NUM_RNG | 8
MAX_SAMPLE  | 4
TIME_PAR | 1

Output |  value
:-------- | :--- 
call price| 6.048
put price | 10.65

For the Asian option,

Input parameter |  value
:-------- | :---
T| 10
S0  | 100
K 	| 105
rate    |  1%
volatility | 15%
MAX_NUM_RNG | 2
MAX_SAMPLE  | 2
TIME_PAR	| 128

Output |  value
:-------- | :--- 
call price| 24.86
put price | 0.33

## Performance Metrics

As discussed above, the computational cost $C=MN$ is a key factor that affects both the performance of the simulation and the quality of the result. The time complexity of the algorithm is $O(C)$, so that we analyze the performance as the total simulation time per step, which is defined as:

$$t=T_s/C$$

For the algorithms in this repository, $t\approx1.25ns$ for an implementation with 8 RNGs. 


[Black-Scholes Model]: https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model 
[geometric Brownian motion]: https://en.wikipedia.org/wiki/Geometric_Brownian_motion	
[Wiener process]: https://en.wikipedia.org/wiki/Wiener_process
[Call options]: https://en.wikipedia.org/wiki/Call_option
[put options]: https://en.wikipedia.org/wiki/Put_option
[Mersenne Twister]: https://en.wikipedia.org/wiki/Mersenne_Twister
[Monte Carlo]: https://en.wikipedia.org/wiki/Monte_Carlo_method  
[Box Muller transformation]: https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform

