#include "cuda_equation.hpp"
#include "cuda_param.hpp"
#include "params.hpp"

#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/device_vector.h>

__device__ long c;
__constant__ static long ic;
__constant__ static long jc;
__constant__ static long kc;

__constant__ static int i_0;
__constant__ static int j_0;
__constant__ static int k_0;

__constant__ static double hx;
__constant__ static double hy;
__constant__ static double hz;
__constant__ static double ht;

__constant__ static long incsize;

void cuda_init(dvector& d_arrayPrev,
               dvector& d_arrayCurr,
               dvector& d_arrayNext,
               long _incsize,
               long _ic,
               long _jc,
               long _kc,
               int _i_0,
               int _j_0,
               int _k_0,
               double _hx,
               double _hy,
               double _hz,
               double _ht) {
  cudaMemcpyToSymbol(incsize, &_incsize, sizeof(incsize), 0,
                     cudaMemcpyHostToDevice);

  cudaMemcpyToSymbol(ic, &_ic, sizeof(ic), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(jc, &_jc, sizeof(jc), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(kc, &_kc, sizeof(kc), 0, cudaMemcpyHostToDevice);

  cudaMemcpyToSymbol(i_0, &_i_0, sizeof(i_0), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(j_0, &_j_0, sizeof(j_0), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(k_0, &_k_0, sizeof(k_0), 0, cudaMemcpyHostToDevice);

  cudaMemcpyToSymbol(hx, &_hx, sizeof(hx), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(hy, &_hy, sizeof(hy), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(hz, &_hz, sizeof(hz), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(ht, &_ht, sizeof(ht), 0, cudaMemcpyHostToDevice);

  try {
    d_arrayPrev.resize(_incsize);
    d_arrayCurr.resize(_incsize);
    d_arrayNext.resize(_incsize);
  } catch (...) {
    std::cerr << "CAUGHT AN EXCEPTION" << std::endl;
  }
}

long cuda_counter() {return c;}

void cuda_prev_to_next(dvector &arrayPrev, dvector &arrayNext)
{
  arrayNext = arrayPrev;
}

void cuda_append(std::vector<hvector>& host,
                 std::vector<dvector>& device,
                 uint sz) {
  host.push_back(hvector());
  host.back().resize(sz);
  device.push_back(dvector());
  device.back().resize(sz);
}

void cuda_copy_step(dvector& arrayPrev,
                    dvector& arrayCurr,
                    dvector& arrayNext) {
  arrayPrev = arrayCurr;
  arrayCurr = arrayNext;
}

__device__ double phi(double x, double y, double z) {
  return sin(y) * cos(x - M_PI_2) * cos(z - M_PI_2);
}

__device__ double u(double x, double y, double z, double t) {
  return phi(x, y, z) * cos(t);
}

__device__ long index(uint i, uint j, uint k) {
  return (long(i) * (jc + 2) + j) * (kc + 2) + k;
}

__device__ long index2(uint j, uint k) {
  return (j + 1) * (kc + 2) + (k + 1);
}

__device__ double x(uint i) {
  return (i_0 + i) * hx;
}

__device__ double y(uint j) {
  return (j_0 + j) * hy;
}

__device__ double z(uint k) {
  return (k_0 + k) * hz;
}

__device__ double deltaTime(uint n) {
  return n * ht;
}

__device__ double calculateIndex(uint i,
                                 uint j,
                                 uint k,
                                 double* arrayCurr,
                                 double* arrayPrev) {
  atomicAdd(&c, long(1));
  long indexC = index(i, j, k);

  return 2 * arrayCurr[indexC] - arrayPrev[indexC] +
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

struct i_j_k {
  int i, j, k;

  __device__ i_j_k(long offset) {
    long ij = offset / (kc + 2);
    i = ij / (jc + 2);
    j = ij % (jc + 2);
    k = offset % (kc + 2);
  }
};

struct j_k {
  int j, k;

  __device__ j_k(long offset) {
    j = offset / (kc + 2);
    k = offset % (kc + 2);
  }
};

struct i_j {
  int i, j;

  __device__ i_j(long offset) {
    i = offset / (jc + 2);
    j = offset % (jc + 2);
  }
};

struct i_k {
  int i, k;

  __device__ i_k(long offset) {
    i = offset / (kc + 2);
    k = offset % (kc + 2);
  }
};

struct residual_functor {
  uint curr_step;
  double* arrayNext;

  residual_functor(uint step, double* array)
      : curr_step(step), arrayNext(array) {}

  __device__ double operator()(long offset) {
    i_j_k index(offset);
    if (index.i == 0 || index.i == ic + 1 || index.j == 0 ||
        index.j == jc + 1 || index.k == 0 || index.k == kc + 1)
      return 0.0;

    double aSol =
        u(x(index.i - 1), y(index.j - 1), z(index.k - 1), deltaTime(curr_step));
    double residual = aSol - arrayNext[offset];
    return std::abs(residual);
  }
};

__host__ double cuda_residual(uint curr_step, dvector& arrayNext, long size) {
  dvector resVec(arrayNext.size());
  thrust::counting_iterator<int> it(0);
  thrust::transform(it, it + resVec.size(), resVec.begin(),
                    residual_functor(curr_step, arrayNext.data().get()));
  return thrust::reduce(resVec.begin(), resVec.end()) / size;
}

struct initPrev_functor {
  __device__ double operator()(long offset) {
    i_j_k index(offset);
    return phi(x(index.i - 1), y(index.j - 1), z(index.k - 1));
  }
};

__host__ void cuda_initPrev(dvector& arrayPrev) {
  thrust::counting_iterator<int> it(0);
  thrust::transform(it, it + arrayPrev.size(), arrayPrev.begin(),
                    initPrev_functor());
}

struct initCurr_functor {
  double* arrayPrev;

  __host__ initCurr_functor(double* array) : arrayPrev(array) {}

  __device__ double operator()(long offset) {
    i_j_k index(offset);
    return arrayPrev[offset] +
           ht * ht / 2 * (-phi(x(index.i - 1), y(index.j - 1), z(index.k - 1)));
  }
};

__host__ void cuda_initCurr(dvector& arrayPrev, dvector& arrayCurr) {
  thrust::counting_iterator<int> it(0);
  thrust::transform(it, it + arrayCurr.size(), arrayCurr.begin(),
                    initCurr_functor(arrayPrev.data().get()));
}

struct calculateIndex_functor {
  double* arrayCurr;
  double* arrayPrev;
  double* arrayNext;

  __host__ calculateIndex_functor(double* array, double* arrayP, double* arrayN)
      : arrayCurr(array), arrayPrev(arrayP), arrayNext(arrayN) {}

  __device__ double operator()(long offset) {
    i_j_k idx(offset);
    if (idx.i > 1 && idx.i < ic - 2 && idx.j > 1 && idx.j < jc - 2 &&
        idx.k > 1 && idx.k < kc - 2)
      return calculateIndex(idx.i, idx.j, idx.k, arrayCurr, arrayPrev);
    else
      return arrayNext[offset];
  }
};

__device__ bool is_border(i_j_k idx) {
  if (idx.i == 0 || idx.i == ic - 1 || idx.j == 0 || idx.j == jc - 1 ||
      idx.k == 0 || idx.k == kc - 1)
    return false;
  return (idx.i == 1 || idx.i == ic - 2 || idx.j == 1 || idx.j == jc - 2 ||
          idx.k == 1 || idx.k == kc - 2);
}

struct calculateIndexDir_functor {
  double* arrayCurr;
  double* arrayPrev;
  double* arrayNext;

  __host__ calculateIndexDir_functor(double* array,
                                     double* arrayP,
                                     double* arrayN)
      : arrayCurr(array), arrayPrev(arrayP), arrayNext(arrayN) {}

  __device__ double operator()(long offset) {
    i_j_k idx(offset);
    if (is_border(idx))
      return calculateIndex(idx.i, idx.j, idx.k, arrayCurr, arrayPrev);
    return arrayNext[offset];
  }
};

__host__ void cuda_calculateIndex(dvector& d_arrayNext,
                                  dvector& d_arrayCurr,
                                  dvector& d_arrayPrev) {
  thrust::counting_iterator<int> it(0);
  thrust::transform(
      it, it + d_arrayNext.size(), d_arrayNext.begin(),
      calculateIndex_functor(d_arrayCurr.data().get(), d_arrayPrev.data().get(),
                             d_arrayNext.data().get()));
}

void cuda_calculateDir(dvector& d_arrayNext,
                       dvector& d_arrayCurr,
                       dvector& d_arrayPrev) {
  thrust::counting_iterator<int> it(0);
  thrust::transform(it, it + d_arrayNext.size(), d_arrayNext.begin(),
                    calculateIndexDir_functor(d_arrayCurr.data().get(),
                                              d_arrayPrev.data().get(),
                                              d_arrayNext.data().get()));
}

struct edgeX_send_functor {
  int fix_i;
  double* arrayCurr;

  __host__ edgeX_send_functor(int _i, double* array)
      : fix_i(_i), arrayCurr(array) {}

  __device__ double operator()(long offset) {
    j_k idx(offset);
    return arrayCurr[index(fix_i, idx.j, idx.k)];
  }
};

struct edgeY_send_functor {
  int fix_j;
  double* arrayCurr;

  __host__ edgeY_send_functor(int j, double* array)
      : fix_j(j), arrayCurr(array) {}

  __device__ double operator()(long offset) {
    i_k idx(offset);
    return arrayCurr[index(idx.i, fix_j, idx.k)];
  }
};

struct edgeZ_send_functor {
  int fix_k;
  double* arrayCurr;

  __host__ edgeZ_send_functor(int k, double* array)
      : fix_k(k), arrayCurr(array) {}

  __device__ double operator()(long offset) {
    i_j idx(offset);
    return arrayCurr[index(idx.i, idx.j, fix_k)];
  }
};

struct edgeX_recv_functor {
  int fix_i;
  double* v;
  double* arrayCurr;

  __host__ edgeX_recv_functor(int _i, double* array, double* arrayC)
      : fix_i(_i), v(array), arrayCurr(arrayC) {}

  __device__ double operator()(long offset) {
    i_j_k idx(offset);
    if (idx.i == fix_i)
      return v[index2(idx.j, idx.k)];
    else
      return arrayCurr[offset];
  }
};

struct edgeY_recv_functor {
  int fix_j;
  double* v;
  double* def;

  __host__ edgeY_recv_functor(int _j, double* array, double* d)
      : fix_j(_j), v(array), def(d) {}

  __device__ double operator()(long offset) {
    i_j_k idx(offset);
    if (idx.j == fix_j)
      return v[index2(idx.i, idx.k)];
    else
      return def[offset];
  }
};

struct edgeZ_recv_functor {
  int fix_k;
  double* v;
  double* def;

  __host__ edgeZ_recv_functor(int _k, double* array, double* d)
      : fix_k(_k), v(array), def(d) {}

  __device__ double operator()(long offset) {
    i_j_k idx(offset);
    if (idx.k == fix_k)
      return v[index2(idx.i, idx.j)];
    else
      return def[offset];
  }
};

void cuda_edgeX(ExchangeDir cdir,
                dvector& v,
                uint id,
                bool recv,
                dvector& d_arrayNext,
                dvector& d_arrayCurr,
                long ic,
                long jc,
                long kc) {
  int i;
  switch (cdir) {
    case plus_x: {
      i = recv ? ic - 1 : ic - 2;
      break;
    }
    case minus_x: {
      i = recv ? 0 : 1;
      break;
    }
    case period_plus_x: {
      i = recv ? ic - 1 : ic - 2;
      break;
    }
    case period_minus_x: {
      i = recv ? 1 : 2;
      break;
    }
    default:
      std::cerr << "BAD CASE" << std::endl;
      exit(1);
  }

  thrust::counting_iterator<int> it(0);

  if (!recv) {
    thrust::transform(it, it + v.size(), v.begin(),
                      edgeX_send_functor(i, d_arrayCurr.data().get()));
  } else {
    thrust::transform(it, it + v.size(), v.begin(),
                      edgeX_recv_functor(i, d_arrayCurr.data().get(),
                                         d_arrayCurr.data().get()));
  }
}

void cuda_edgeY(ExchangeDir cdir,
                dvector& v,
                uint id,
                bool recv,
                dvector& d_arrayNext,
                dvector& d_arrayCurr,
                long jc) {
  int j;
  switch (cdir) {
    case plus_y: {
      j = recv ? jc - 1 : jc - 2;
      break;
    }
    case minus_y: {
      j = recv ? 0 : 1;
      break;
    }
    default:
      std::cerr << "BAD CASE" << std::endl;
      exit(1);
  }

  thrust::counting_iterator<int> it(0);

  if (!recv) {
    thrust::transform(it, it + v.size(), v.begin(),
                      edgeY_send_functor(j, d_arrayCurr.data().get()));
  } else {
    thrust::transform(it, it + v.size(), v.begin(),
                      edgeY_recv_functor(j, d_arrayCurr.data().get(),
                                         d_arrayCurr.data().get()));
  }
}

void cuda_edgeZ(ExchangeDir cdir,
                dvector& v,
                uint id,
                bool recv,
                dvector& d_arrayNext,
                dvector& d_arrayCurr,
                long ic,
                long jc,
                long kc) {
  int k;
  switch (cdir) {
    case plus_z: {
      k = recv ? kc - 1 : kc - 2;
      break;
    }
    case minus_z: {
      k = recv ? 0 : 1;
      break;
    }
    case period_plus_z: {
      k = recv ? kc - 1 : kc - 2;
      break;
    }
    case period_minus_z: {
      k = recv ? 1 : 2;
      break;
    }
    default:
      std::cerr << "BAD CASE" << std::endl;
      exit(1);
  }
  thrust::counting_iterator<int> it(0);

  if (!recv) {
    thrust::transform(it, it + v.size(), v.begin(),
                      edgeZ_send_functor(k, d_arrayCurr.data().get()));
  } else {
    thrust::transform(it, it + v.size(), v.begin(),
                      edgeZ_recv_functor(k, d_arrayCurr.data().get(),
                                         d_arrayCurr.data().get()));
  }
}
