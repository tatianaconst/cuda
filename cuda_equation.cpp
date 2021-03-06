#include "cuda_equation.hpp"
#include <string.h>
#include "cuda_runtime.h"

ProcessorNode node;
Reporter r;

Equation::Equation(uint n) : N(n), curr_step(0) {}

void Equation::init() {
  double t_init_start = MPI_Wtime();

  ic = N / node.PhysSize.x;
  jc = N / node.PhysSize.y;
  kc = N / node.PhysSize.z;

  K = 20;
  T = 0.01;

  hx = Lx / ic;
  hy = Ly / jc;
  hz = Lz / kc;
  ht = T / K;

  if (node.x < N % node.PhysSize.x)
    ++ic;
  if (node.y < N % node.PhysSize.y)
    ++jc;
  if (node.z < N % node.PhysSize.z)
    ++kc;

  int it = N % node.PhysSize.x;
  int jt = N % node.PhysSize.y;
  int kt = N % node.PhysSize.z;

  i0 = std::min(node.x, it) * (ic + 1) + std::max(node.x - it, 0) * ic;
  j0 = std::min(node.y, jt) * (jc + 1) + std::max(node.y - jt, 0) * jc;
  k0 = std::min(node.z, kt) * (kc + 1) + std::max(node.z - kt, 0) * kc;

  incsize = (ic + 2) * (jc + 2) * (kc + 2);
  cuda_init(d_arrayPrev, d_arrayCurr, d_arrayNext, incsize, ic, jc, kc, i0, j0,
            k0, hx, hy, hz, ht);

  for (int i = 0; i < 10; ++i) {
    ExchangeDir cdir = static_cast<ExchangeDir>(i);
    if (node.is(cdir)) {
      uint size;
      if (cdir == plus_x || cdir == minus_x || cdir == period_plus_x ||
          cdir == period_minus_x)
        size = jc * kc;
      else if (cdir == plus_y || cdir == minus_y)
        size = ic * kc;
      else
        size = ic * jc;
      send_requests.append(cdir, size);
      recv_requests.append(cdir, size);
    }
  }
  double t_init_finish = MPI_Wtime();
  r.t_init += t_init_finish - t_init_start;
}

double Equation::run() {


  double t_calculation_start = MPI_Wtime();
  cuda_initPrev(d_arrayPrev);
  //d_arrayNext = d_arrayPrev;
  cuda_prev_to_next(d_arrayPrev, d_arrayNext);
  cuda_initCurr(d_arrayPrev, d_arrayCurr);
  double t_calculation_finish = MPI_Wtime();
  r.t_calculation += t_calculation_finish - t_calculation_start;


  uint curr_step = 2;
  double totalSumRes = 0;

  for (; curr_step < K + 1; ++curr_step) {
    //std::cout << curr_step;
    for (uint i = 0; i < recv_requests.size(); ++i) {
      if (curr_step != 2)
        MPI_Irecv(recv_requests.host[i].data(), recv_requests.host[i].size(),
                  MPI_DOUBLE, node.neighbor(recv_requests.iv[i]),
                  pairDir(recv_requests.iv[i]), MPI_COMM_WORLD,
                  &recv_requests.v[i]);
    }


    t_calculation_start = MPI_Wtime();
    cuda_calculateIndex(d_arrayNext, d_arrayCurr, d_arrayPrev);
    t_calculation_finish = MPI_Wtime();
    r.t_calculation += t_calculation_finish - t_calculation_start;


    r.timer_send = MPI_Wtime();
    MPI_Waitall(recv_requests.v.size(), recv_requests.v.data(),
                MPI_STATUSES_IGNORE);
    double t_send_fin = MPI_Wtime();
    r.t_MPI += t_send_fin - r.timer_send;



    double t_host_device_start = MPI_Wtime();
    recv_requests.cpu_to_gpu();
    double t_host_device_finish = MPI_Wtime();
    r.t_host_device += t_host_device_finish - t_host_device_start;


    t_calculation_start = MPI_Wtime();
    for (uint i = 0; i < recv_requests.size(); ++i)
      copy(recv_requests, i, true);
    t_calculation_finish = MPI_Wtime();
    r.t_calculation += t_calculation_finish - t_calculation_start;


    t_calculation_start = MPI_Wtime();
    cuda_calculateDir(d_arrayNext, d_arrayCurr, d_arrayPrev);
    t_calculation_finish = MPI_Wtime();
    r.t_calculation += t_calculation_finish - t_calculation_start;


    if (curr_step != K) {
      for (int i = 0; i < send_requests.size(); ++i) {

        t_calculation_start = MPI_Wtime();
        copy(send_requests, i, false);
        t_calculation_finish = MPI_Wtime();
        r.t_calculation += t_calculation_finish - t_calculation_start;


        r.t_send = MPI_Wtime();
        MPI_Isend(send_requests.host[i].data(), send_requests.host[i].size(),
                  MPI_DOUBLE, node.neighbor(send_requests.iv[i]),
                  send_requests.iv[i], MPI_COMM_WORLD, &send_requests.v[i]);
      }


      t_host_device_start = MPI_Wtime();
      send_requests.gpu_to_cpu();
      t_host_device_finish = MPI_Wtime();
      r.t_host_device += t_host_device_finish - t_host_device_start;


      t_calculation_start = MPI_Wtime();
      for (int i = 0; i < send_requests.size(); ++i)
        copy(send_requests, i, false);
      t_calculation_finish = MPI_Wtime();
      r.t_calculation += t_calculation_finish - t_calculation_start;
      
      
    }


      t_calculation_start = MPI_Wtime();
      if (curr_step == K)
        totalSumRes = cuda_residual(curr_step, d_arrayNext, N * N * N);
      t_calculation_finish = MPI_Wtime();
      r.t_calculation += t_calculation_finish - t_calculation_start;
    

    if (curr_step < K)
      cuda_copy_step(d_arrayPrev, d_arrayCurr, d_arrayNext);
  }

  double sharedResidual = 0;
  MPI_Barrier(MPI_COMM_WORLD);

  double t_MPI_start = MPI_Wtime();
  MPI_Reduce(&totalSumRes, &sharedResidual, 1, MPI_DOUBLE, MPI_SUM, 0,
             MPI_COMM_WORLD);
  double t_MPI_finish = MPI_Wtime();
  r.t_MPI += t_MPI_finish - t_MPI_start;

  std::cout << cuda_counter() << std::endl;

  return sharedResidual / r.cpu;
}

void Equation::copy(ProcessorNode::Requests& requests, uint id, bool recv) {
  inRange(id, 0, requests.iv.size());

  ExchangeDir cdir = requests.iv[id];
  switch (cdir) {
    case plus_x:
    case minus_x:
    case period_plus_x:
    case period_minus_x: {
      cuda_edgeX(requests.iv[id], requests.device[id], id, recv, d_arrayNext,
                 d_arrayCurr, ic, jc, kc);
      break;
    }
    case plus_y:
    case minus_y: {
      cuda_edgeY(requests.iv[id], requests.device[id], id, recv, d_arrayNext,
                 d_arrayCurr, jc);
      break;
    }
    case plus_z:
    case minus_z:
    case period_plus_z:
    case period_minus_z: {
      cuda_edgeZ(requests.iv[id], requests.device[id], id, recv, d_arrayNext,
                 d_arrayCurr, ic, jc, kc);
      break;
    }
    default:
      assert(false);
  }
}

void Equation::inRange(int i, int a, int b) {
  assert((i >= a) && (i < b));
}

void ProcessorNode::init() {
  double t_init_start = MPI_Wtime();
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  size = world_size;
  r.cpu = world_size;

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  rank = world_rank;

  PhysSize = getPhysGrid(size);
  z = rank % PhysSize.z;
  y = rank / PhysSize.z % PhysSize.y;
  x = rank / PhysSize.z / PhysSize.y;
  double t_init_finish = MPI_Wtime();
  r.t_init += t_init_finish - t_init_start;


  cudaGetDeviceCount(&gpus);
  cudaSetDevice(world_size % gpus);
}

int ProcessorNode::neighbor(ExchangeDir cdir) const {
  switch (cdir) {
    case plus_x:
      return toRank(x + 1, y, z);
    case minus_x:
      return toRank(x - 1, y, z);
    case plus_y:
      return toRank(x, y + 1, z);
    case minus_y:
      return toRank(x, y - 1, z);
    case plus_z:
      return toRank(x, y, z + 1);
    case minus_z:
      return toRank(x, y, z - 1);
    case period_plus_x:
      return toRank(x + 1 - PhysSize.x, y, z);
    case period_minus_x:
      return toRank(x + PhysSize.x - 1, y, z);
    case period_plus_z:
      return toRank(x, y, z + 1 - PhysSize.z);
    case period_minus_z:
      return toRank(x, y, z + PhysSize.z - 1);
    default:
      return -1;
  }
}

bool ProcessorNode::is(ExchangeDir cdir) const {
  return neighbor(cdir) != -1;
}

int ProcessorNode::toRank(uint i, uint j, uint k) const {
  if (i >= PhysSize.x || j >= PhysSize.y || k >= PhysSize.z) {
    return -1;
  }
  return (i * PhysSize.y + j) * PhysSize.z + k;
}

void ProcessorNode::Requests::append(ExchangeDir dir, uint sz) {
  iv.push_back(dir);
  v.push_back(MPI_REQUEST_NULL);

  cuda_append(host, device, sz);
}
