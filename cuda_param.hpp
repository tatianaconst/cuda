#ifndef CUDA_PARAM_HPP
#define CUDA_PARAM_HPP

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

typedef thrust::device_vector<float> dvector;
typedef thrust::host_vector<float> hvector;



#include "params.hpp"

// float x(uint i); 

// float y(uint j); 

// float z(uint k); 

// void calculateIndex(uint i, uint j, uint k, float *indexCurr, float *arrayPrev); 

void cuda_init(dvector &d_arrayPrev,
			   dvector &d_arrayCurr,
			   dvector &d_arrayNext,
			   long incsize);
void cuda_append(std::vector<hvector> &host, std::vector<dvector> &device, uint sz);

void cuda_copy_step(dvector &arrayPrev, dvector &arrayCurr, dvector &arrayNext);

float cuda_residual(uint curr_step, dvector arrayNext);

void cuda_initPrev(dvector &arrayPrev);


void cuda_initCurr(dvector &arrayPrev, dvector &arrayCurr);

void cuda_calculateIndex(dvector &d_arrayNext, dvector &d_arrayCurr, 
						 dvector &d_arrayPrev);

void cuda_calculateDir(dvector &d_arrayNext, dvector &d_arrayCurr, 
						 dvector &d_arrayPrev);

void cuda_edgeX(ExchangeDir cdir, dvector &v, uint id, bool recv, 
				dvector d_arrayNext, dvector d_arrayCurr);

void cuda_edgeY(ExchangeDir cdir, dvector &v, uint id, bool recv, 
				dvector d_arrayNext, dvector d_arrayCurr);

void cuda_edgeZ(ExchangeDir cdir, dvector &v, uint id, bool recv, 
				dvector d_arrayNext, dvector d_arrayCurr);


#endif 