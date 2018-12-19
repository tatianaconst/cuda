#ifndef CUDA_PARAM_HPP
#define CUDA_PARAM_HPP

#include "params.hpp"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

typedef thrust::device_vector<double> dvector;
typedef thrust::host_vector<double> hvector;



void cuda_init(dvector &d_arrayPrev,
			   dvector &d_arrayCurr,
			   dvector &d_arrayNext,
			   long _incsize, long _ic, long _jc, long _kc,
			   int _i_0, int _j_0, int _k_0,
               double _hx, double _hy, double _hz, double _ht);

void cuda_append(std::vector<hvector> &host, std::vector<dvector> &device, uint sz);

void cuda_copy_step(dvector &arrayPrev, dvector &arrayCurr, dvector &arrayNext);

double cuda_residual(uint curr_step, dvector &arrayNext, long size);

void cuda_initPrev(dvector &arrayPrev);


void cuda_initCurr(dvector &arrayPrev, dvector &arrayCurr);

void cuda_calculateIndex(dvector &d_arrayNext, dvector &d_arrayCurr, 
						 dvector &d_arrayPrev);

void cuda_calculateDir(dvector &d_arrayNext, dvector &d_arrayCurr, 
						 dvector &d_arrayPrev);

void cuda_edgeX(ExchangeDir cdir, dvector &v, uint id, bool recv, 
				dvector &d_arrayNext, dvector &d_arrayCurr, long ic, long jc, long kc);

void cuda_edgeY(ExchangeDir cdir, dvector &v, uint id, bool recv, 
				dvector &d_arrayNext, dvector &d_arrayCurr, long jc);

void cuda_edgeZ(ExchangeDir cdir, dvector &v, uint id, bool recv, 
				dvector &d_arrayNext, dvector &d_arrayCurr, long ic, long jc, long kc);


#endif 
