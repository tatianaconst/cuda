#include "equation.hpp"
#include <string.h>

ProcessorNode node;

Equation::Equation(uint n) : N(n), curr_step(0), maxResidual(0) {}

void Equation::init() {
  ic = N;
  jc = N;
  kc = N;

  hx = Lx / ic;
  hy = Ly / jc;
  hz = Lz / kc;
  ht = T / K;

  incsize = (ic + 2) * (jc + 2) * (kc + 2);
  arrayPrev.resize(incsize);
  arrayCurr.resize(incsize);
  arrayNext.resize(incsize);
}

void Equation::run() {
  initPrev();
  initCurr();
  for (; curr_step < K + 1; ++curr_step) {
#pragma omp parallel for
    for (int i = 1; i < ic - 1; ++i) {
      for (uint j = 1; j < jc - 1; ++j) {
        for (uint k = 1; k < kc - 1; ++k) {
          calculateIndex(i, j, k);
        }
      }
    }

#pragma omp parallel for
    for (int i = 0; i < ic; ++i) {
      for (uint j = 0; j < jc; ++j) {
        for (uint k = 0; k < kc; ++k) {
          long id = index(i, j, k);
          double aSol = u(x(i), y(j), z(k), deltaTime(curr_step));
          double residual = aSol - arrayNext[id];
          sumResidual += std::abs(residual);
        }
      }
    }
    if (curr_step < K) {
      memcpy(arrayPrev.data(), arrayCurr.data(), incsize * sizeof(double));
      memcpy(arrayCurr.data(), arrayNext.data(), incsize * sizeof(double));
    }
  }
  sumResidual = sumResidual / (ic * jc * kc);
  printf("Np: %d, N: %d, R: %lf\n", node.size, N, sumResidual);
}

double Equation::x(uint i) const 
{ 
  return (i0 + i) * hx; 
}

double Equation::y(uint j) const 
{ 
  return (j0 + j) * hy; 
}

double Equation::z(uint k) const 
{ 
  return (k0 + k) * hz; 
}

void Equation::inRange(int i, int a, int b) 
{ 
  assert((i >= a) && (i < b)); 
}

void Equation::calculateIndex(uint i, uint j, uint k) {
  long indexC = index(i, j, k);
  inRange(indexC, 0, arrayNext.size());
  inRange(indexC, 0, arrayNext.size());
  inRange(index(i - 1, j, k), 0, arrayNext.size());
  inRange(index(i + 1, j, k), 0, arrayNext.size());
  inRange(index(i, j - 1, k), 0, arrayNext.size());
  inRange(index(i, j + 1, k), 0, arrayNext.size());
  inRange(index(i, j, k - 1), 0, arrayNext.size());
  inRange(index(i, j, k + 1), 0, arrayNext.size());
  arrayNext[indexC] =
      2 * arrayCurr[indexC] - arrayPrev[indexC] +
      ht * ht *
          ((arrayCurr[index(i - 1, j, k)] - 2 * arrayCurr[indexC] +
            arrayCurr[index(i + 1, j, k)]) /
               hx / hx +
           (arrayCurr[index(i, j - 1, k)] - 2 * arrayCurr[indexC] +
            arrayCurr[index(i, j + 1, k)]) /
               hy / hy +
           (arrayCurr[index(i, j, k - 1)] - 2 * arrayCurr[indexC] +
            arrayCurr[index(i, j, k + 1)]) /
               hz / hz);
}

uint Equation::index(uint i, uint j, uint k) const {
  return ((i + 1) * (jc + 2) + (j + 1)) * (kc + 2) + (k + 1);
}

void Equation::initPrev() 
{
  for (int i = 0; i < ic; ++i)
    for (uint j = 0; j < jc; ++j)
      for (uint k = 0; k < kc; ++k) {
        uint indexC = index(i, j, k);
        inRange(indexC, 0, arrayPrev.size());
        arrayPrev[indexC] = phi(x(i), y(j), z(k));
      }
}
void Equation::initCurr() 
{
  for (int i = 0; i < ic; ++i)
    for (uint j = 0; j < jc; ++j)
      for (uint k = 0; k < kc; ++k) {
        uint indexC = index(i, j, k);
        inRange(indexC, 0, arrayCurr.size());
        arrayCurr[indexC] =
            arrayPrev[indexC] + ht * ht / 2 * (-phi(x(i), y(j), z(k)));
      }
  curr_step = 2;
}


//
