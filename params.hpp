#ifndef PARAMS_HPP
#define PARAMS_HPP

#include <omp.h>

#include <algorithm>
#include <assert.h>
#include <cmath>
#include <mpi.h>
#include <stdio.h>
#include <vector>

void print(int rank, const char *str);
void print(int rank, const char *str, int n);


struct PhysSize {
  uint x, y, z;
  PhysSize(uint x0 = 0, uint y0 = 0, uint z0 = 0) : x(x0), y(y0), z(z0) {}
};

typedef std::vector<float> rvector;

PhysSize getPhysGrid(uint i);

enum ExchangeDir {
  plus_x,
  minus_x,
  plus_y,
  minus_y,
  plus_z,
  minus_z,
  period_plus_x,
  period_minus_x,
  period_plus_z,
  period_minus_z
};

// extern double T;  // = 0.01;
extern double Lx; // = M_PI;
extern double Ly; // = M_PI;
extern double Lz; // = M_PI;
// extern int K;  // = 20;

ExchangeDir pairDir(ExchangeDir cdir);

#endif
