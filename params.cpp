#include "params.hpp"

double Lx = M_PI;
double Ly = M_PI;
double Lz = M_PI;

PhysSize getPhysGrid(uint i) {
  switch (i) {
    case 1:
      return PhysSize(1, 1, 1);
    case 2:
      return PhysSize(2, 1, 1);
    case 4:
      return PhysSize(2, 2, 1);
    case 8:
      return PhysSize(2, 2, 2);
    case 16:
      return PhysSize(4, 2, 2);
    case 32:
      return PhysSize(4, 4, 2);
    case 128:
      return PhysSize(4, 4, 8);
    case 256:
      return PhysSize(8, 4, 8);
    case 512:
      return PhysSize(8, 8, 8);
    default:
      return PhysSize(1, 1, 1);
  }
}

ExchangeDir pairDir(ExchangeDir cdir) {
  switch (cdir) {
    case plus_x:
      return minus_x;
    case minus_x:
      return plus_x;
    case plus_y:
      return minus_y;
    case minus_y:
      return plus_y;
    case plus_z:
      return minus_z;
    case minus_z:
      return plus_z;
    case period_plus_x:
      return period_minus_x;
    case period_minus_x:
      return period_plus_x;
    case period_plus_z:
      return period_minus_z;
    case period_minus_z:
      return period_plus_z;
    default:
      assert(false);
  }
}

void print(int rank, const char* str) {
  if (rank == 0);
  // printf(str);
}

void print(int rank, const char* str, int n) {
  if (rank == 0);
  // printf(str, n);
}
