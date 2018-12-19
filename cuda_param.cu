#include "cuda_param.hpp"
#include "params.hpp"
#include "cuda_equation.hpp"



__constant__ static long ic;
__constant__ static long jc;
__constant__ static long kc;

__constant__ static int i_0;
__constant__ static int j_0;
__constant__ static int k_0;

__constant__ static float hx;
__constant__ static float hy;
__constant__ static float hz;
__constant__ static float ht;

__constant__ static long incsize;


void cuda_init(dvector &d_arrayPrev,
			   dvector &d_arrayCurr,
			   dvector &d_arrayNext,
			   long _incsize, long _ic, long _jc, long _kc,
			   int _i_0, int _j_0, int _k_0,
			   float _hx, float _hy, float _hz, float _ht)
{
	cudaMemcpyToSymbol(incsize, &_incsize, sizeof(incsize), 0, cudaMemcpyHostToDevice);

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
	}
    catch(...) {
        std::cerr << "CAUGHT AN EXCEPTION" << std::endl;
    }
    std::cout << "Resize OK" << std::endl;
}

void cuda_append(std::vector<hvector> &host, std::vector<dvector> &device, uint sz)
{
	host.push_back(hvector());
	host.back().resize(sz);
	device.push_back(dvector());
	device.back().resize(sz);
}

void cuda_copy_step(dvector &arrayPrev, dvector &arrayCurr, dvector &arrayNext)
{
	arrayPrev = arrayCurr;
	arrayCurr = arrayNext;
}



__device__
float phi(double x, double y, double z) 
{
  return sin(y) * cos(x - M_PI_2) * cos(z - M_PI_2);
}


__device__
float u(double x, double y, double z, double t) 
{
  return phi(x, y, z) * cos(t);
}


__device__
long index(uint i, uint j, uint k)
{
	return (long(i) * (jc + 2) + j) * (kc + 2) + k;
}

__device__
long index2(uint j, uint k)
{
  return  (j + 1) * (kc + 2) + (k + 1);
}

__device__
float x(uint i)
{ 
  return (i_0 + i) * hx; 
}

__device__
float y(uint j)
{ 
  return (j_0 + j) * hy; 
}

__device__
float z(uint k)
{ 
  return (k_0 + k) * hz; 
}

__device__
float deltaTime(uint n) 
{ 
	return n * ht; 
}

__device__
float calculateIndex(uint i, uint j, uint k, 
					float *arrayCurr,
					float *arrayPrev) 
{
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


struct i_j_k
{
	int i, j, k;

	__device__
	i_j_k (long offset)
	{
		long ij = offset / (kc + 2);
		i = ij / (jc + 2);
		j = ij % (jc + 2);
		k = offset % (kc + 2);
	}
};

struct j_k
{
	int j, k;

	__device__
	j_k (long offset)
	{
		j = offset / (kc + 2);
		// i = ij / (jc + 2);
		// j = ij % (jc + 2);
		k = offset % (kc + 2);
	}
};

struct i_j
{
	int i, j;

	__device__
	i_j (long offset)
	{
		i = offset / (jc + 2);
		// i = ij / (jc + 2);
		// j = ij % (jc + 2);
		j = offset % (jc + 2);
	}
};

struct i_k
{
	int i, k;

	__device__
	i_k (long offset)
	{
		i = offset / (kc + 2);
		// i = ij / (jc + 2);
		// j = ij % (jc + 2);
		k = offset % (kc + 2);
	}
};

struct residual_functor
{
	uint curr_step;
	float *arrayNext;

	residual_functor(uint step, float *array)
	:curr_step(step), arrayNext(array)
	{}

	__device__
	float operator()(long offset)
	{
		i_j_k index(offset);
		if (index.i == 0 || index.i == ic + 1 ||
			index.j == 0 || index.j == jc + 1 ||
			index.k == 0 || index.k == kc + 1)
			return 0.0;

		float aSol = u(x(index.i), y(index.j), z(index.k), 
					   deltaTime(curr_step));
		float residual = aSol - arrayNext[offset];
		return std::abs(residual);		
	}
};

__host__
float cuda_residual(uint curr_step, dvector arrayNext)
{
	dvector resVec(arrayNext.size());
	thrust::counting_iterator<int> it(0);
	thrust::transform(it, it + resVec.size(), resVec.begin(),
					  residual_functor(curr_step, arrayNext.data().get()));
	return thrust::reduce(resVec.begin(), resVec.end()) / resVec.size();
	// return thrust::inclusive_scan(arrayNext.begin(), arrayNext.end(),
	// 							  float(0.0),
	// 					   		  residual_functor(curr_step)
	// 					   		 );
}



struct initPrev_functor
{
	__device__
	float operator()(long offset) 
	{
		i_j_k index(offset);
		return phi(x(index.i), y(index.j), z(index.k));
	}
};

__host__
void cuda_initPrev(dvector &arrayPrev)
{
	thrust::counting_iterator<int> it(0);
	thrust::transform(it, it + arrayPrev.size(), 
					  arrayPrev.begin(), initPrev_functor());
}


struct initCurr_functor
{
	float *arrayPrev;

	__host__
	initCurr_functor(float *array)
	:arrayPrev(array)
	{}

	__device__
	float operator()(long offset)
	{
		i_j_k index(offset);
		return arrayPrev[offset] + ht * ht / 2 * 
			   (-phi(x(index.i), y(index.j), z(index.k)));
	}
};

__host__
void cuda_initCurr(dvector &arrayPrev, dvector &arrayCurr)
{
	thrust::counting_iterator<int> it;
	thrust::transform(it, it + arrayCurr.size(), arrayCurr.begin(),
					  initCurr_functor(arrayPrev.data().get()));
}

struct calculateIndex_functor
{
	float *arrayCurr;
	float *arrayPrev;
	float *arrayNext;

	__host__
	calculateIndex_functor(float *array, float *arrayP, float *arrayN)
	:arrayCurr(array), arrayPrev(arrayP), arrayNext(arrayN)
	{}

	__device__
	float operator()(long offset)
	{
		i_j_k idx(offset);
		if (idx.i > 1 && idx.i < ic - 2 && 
			idx.j > 1 && idx.j < jc - 2 && 
			idx.k > 1 && idx.k < kc - 2)
			return calculateIndex(idx.i, idx.j, idx.k, arrayCurr, arrayPrev);
		else
			return arrayNext[offset];
	}
};

struct calculateIndexDir_functor
{
	float *arrayCurr;
	float *arrayPrev;
	float *arrayNext;


	__host__
	calculateIndexDir_functor(float *array, float *arrayP, float *arrayN)
	:arrayCurr(array), arrayPrev(arrayP), arrayNext(arrayN)
	{}

	__device__
	float operator()(long offset)
	{
		i_j_k idx(offset);
		if (idx.i == 1 || idx.i == ic - 2 || 
			idx.j == 1 || idx.j == jc - 2 || 
			idx.k == 1 || idx.k == kc - 2)
			return calculateIndex(idx.i, idx.j, idx.k, arrayCurr, arrayPrev);
		else return arrayNext[offset];
	}
};


__host__
void cuda_calculateIndex(dvector &d_arrayNext, dvector &d_arrayCurr, 
						 dvector &d_arrayPrev)
{
	thrust::counting_iterator<int> it;
	thrust::transform(it, it + d_arrayNext.size(), d_arrayNext.begin(),
					  calculateIndex_functor(d_arrayCurr.data().get(),
					  						 d_arrayPrev.data().get(),
					  						 d_arrayNext.data().get()));
}

void cuda_calculateDir(dvector &d_arrayNext, dvector &d_arrayCurr, 
						 dvector &d_arrayPrev)
{
	thrust::counting_iterator<int> it;
	thrust::transform(it, it + d_arrayNext.size(), d_arrayNext.begin(),
					  calculateIndexDir_functor(d_arrayCurr.data().get(),
					  						 d_arrayPrev.data().get(), d_arrayNext.data().get()));
}

struct edgeX_send_functor
{
	int fix_i;
	float *arrayCurr;

	__host__
	edgeX_send_functor(int _i, float *array)
	:fix_i(_i), arrayCurr(array)
	{}

	__device__
	float operator()(long offset)
	{
		j_k idx(offset);
		return arrayCurr[index(fix_i, idx.j, idx.k)];
	}

};

struct edgeY_send_functor
{
	int fix_j;
	float *arrayCurr;

	__host__
	edgeY_send_functor(int j, float *array)
	:fix_j(j), arrayCurr(array)
	{}

	__device__
	float operator()(long offset)
	{
		i_k idx(offset);
		return arrayCurr[index(idx.i, fix_j, idx.k)];
	}

};

struct edgeZ_send_functor
{
	int fix_k;
	float *arrayCurr;

	__host__
	edgeZ_send_functor(int k, float *array)
	:fix_k(k), arrayCurr(array)
	{}

	__device__
	float operator()(long offset)
	{
		i_j idx(offset);
		return arrayCurr[index(idx.i, idx.j, fix_k)];
	}

};

struct edgeX_recv_functor
{
	int fix_i;
	float *v;
	float *arrayCurr;

	__host__
	edgeX_recv_functor(int _i, float *array, float *arrayC)
	:fix_i(_i), v(array), arrayCurr(arrayC)
	{}

	__device__
	float operator()(long offset)
	{
		i_j_k idx(offset);
		if (idx.i == fix_i)
			return v[index2(idx.j, idx.k)];
		else 
			return arrayCurr[offset];
	}
};

struct edgeY_recv_functor
{
	int fix_j;
	float *v;
	float *def;

	__host__
	edgeY_recv_functor(int _j, float *array, float *d)
	:fix_j(_j), v(array), def(d)
	{}

	__device__
	float operator()(long offset)
	{
		i_j_k idx(offset);
		if (idx.j == fix_j)
			return v[index2(idx.i, idx.k)];
		else
			return def[offset];
	}
};

struct edgeZ_recv_functor
{
	int fix_k;
	float *v;
	float *def;

	__host__
	edgeZ_recv_functor(int _k, float *array, float *d)
	:fix_k(_k), v(array), def(d)
	{}

	__device__
	float operator()(long offset)
	{
		i_j_k idx(offset);
		if (idx.k == fix_k)
			return v[index2(idx.i, idx.j)];
		else
			return def[offset];
	}
};


void cuda_edgeX(ExchangeDir cdir, dvector &v, uint id, bool recv, 
				dvector &d_arrayNext, dvector &d_arrayCurr, long ic, long jc, long kc) 
{
  // ExchangeDir cdir = requests.iv[id];
  // dvector &v = requests.device[id];
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
  }

  	thrust::counting_iterator<int> it;

	if (!recv) {
		thrust::transform(it, it + v.size(), v.begin(),
					  edgeX_send_functor(i, d_arrayCurr.data().get()));
	    // v[offset++] = d_arrayCurr[index(i, j, k)];
	}
	else {
		thrust::transform(it, it + v.size(), v.begin(),
					  edgeX_recv_functor(i, d_arrayCurr.data().get(), d_arrayCurr.data().get()));
		//d_arrayCurr[index(i, j, k)] = v[offset++];
	}
}

void cuda_edgeY(ExchangeDir cdir, dvector &v, uint id, bool recv, 
				dvector &d_arrayNext, dvector &d_arrayCurr, long jc) {
  // ExchangeDir cdir = requests.iv[id];
  // thrust::host_vector<float> &v = requests.host[id];
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
  }

  thrust::counting_iterator<int> it;

	if (!recv) {
		thrust::transform(it, it + v.size(), v.begin(),
					  edgeY_send_functor(j, d_arrayCurr.data().get()));
	    // v[offset++] = d_arrayCurr[index(i, j, k)];
	}
	else {
		thrust::transform(it, it + v.size(), v.begin(),
					  edgeY_recv_functor(j, d_arrayCurr.data().get(), d_arrayCurr.data().get()));
		//d_arrayCurr[index(i, j, k)] = v[offset++];
	}
}

void cuda_edgeZ(ExchangeDir cdir, dvector &v, uint id, bool recv, 
				dvector &d_arrayNext, dvector &d_arrayCurr, long ic, long jc, long kc) {
  // ExchangeDir cdir = requests.iv[id];
  // std::vector<float> &v = requests.host[id];
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
  }
    thrust::counting_iterator<int> it;

	if (!recv) {
		thrust::transform(it, it + v.size(), v.begin(),
					  edgeZ_send_functor(k, d_arrayCurr.data().get()));
	    // v[offset++] = d_arrayCurr[index(i, j, k)];
	}
	else {
		thrust::transform(it, it + v.size(), v.begin(),
					  edgeZ_recv_functor(k, d_arrayCurr.data().get(), d_arrayCurr.data().get()));
		//d_arrayCurr[index(i, j, k)] = v[offset++];
	}
}


