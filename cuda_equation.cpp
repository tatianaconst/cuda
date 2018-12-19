#include "cuda_equation.hpp"
#include <string.h>

ProcessorNode node;

Equation::Equation(uint n) : N(n), curr_step(0) {}

void Equation::init() {
  ic = N / node.PhysSize.x;
  jc = N / node.PhysSize.y;
  kc = N / node.PhysSize.z;

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

  incsize = (ic + 2) * (jc + 2) * (kc + 2);
  cuda_init(d_arrayPrev, d_arrayCurr, d_arrayNext, incsize);

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
}

void Equation::run() {
  cuda_initPrev(d_arrayPrev);
  d_arrayNext = d_arrayPrev;

  cuda_initCurr(d_arrayPrev, d_arrayCurr);
  uint curr_step = 2;

  float totalSumRes = 0;
  
  for (; curr_step < K + 1; ++curr_step) {
    for (uint i = 0; i < recv_requests.size(); ++i) {
      print(node.rank, "i = %d\n", i);
      MPI_Irecv(recv_requests.host[i].data(), recv_requests.host[i].size(),
                MPI_FLOAT, node.neighbor(recv_requests.iv[i]),
                pairDir(recv_requests.iv[i]), MPI_COMM_WORLD,
                &recv_requests.v[i]);
    }
    for (int i = 0; i < send_requests.size(); ++i) {
      copy(send_requests, i, false);

      MPI_Isend(send_requests.host[i].data(), send_requests.host[i].size(),
                MPI_FLOAT, node.neighbor(send_requests.iv[i]),
                send_requests.iv[i], MPI_COMM_WORLD, &send_requests.v[i]);
    }

    cuda_calculateIndex(d_arrayNext, d_arrayCurr, d_arrayPrev);

    MPI_Waitall(recv_requests.v.size(), recv_requests.v.data(),
                MPI_STATUSES_IGNORE);
    for (uint i = 0; i < recv_requests.size(); ++i)
      copy(recv_requests, i, true);

    recv_requests.cpu_to_gpu();

    cuda_calculateDir(d_arrayNext, d_arrayCurr, d_arrayPrev);
    
    send_requests.gpu_to_cpu();

    totalSumRes += cuda_residual(curr_step, d_arrayNext);

    if (curr_step < K)
      cuda_copy_step(d_arrayPrev, d_arrayCurr, d_arrayNext);
  }
  totalSumRes = totalSumRes / K;
  float sharedResidual = 0;
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Reduce(&totalSumRes, &sharedResidual, 1, MPI_FLOAT, MPI_SUM, 0,
             MPI_COMM_WORLD);
  if (node.rank == 0)
    printf("Np: %d, N: %ld, R: %lf\n", node.size, N, sharedResidual);
}







// void Equation::edgeX(ProcessorNode::Requests &requests, uint id, bool recv) {
//   ExchangeDir cdir = requests.iv[id];
//   thrust::host_vector<float> &v = requests.host[id];
//   uint offset = 0;
//   int i;
//   switch (cdir) {
//   case plus_x: {
//     i = recv ? ic : ic - 1;
//     break;
//   }
//   case minus_x: {
//     i = recv ? -1 : 0;
//     break;
//   }
//   case period_plus_x: {
//     i = recv ? ic : ic - 1;
//     break;
//   }
//   case period_minus_x: {
//     i = recv ? 0 : 1;
//     break;
//   }
//   }
//   thrust::host_vector<float> &a =
//       (((cdir == period_minus_x) && recv) ? d_arrayNext : d_arrayCurr);
//   for (uint j = 0; j < jc; ++j) {
//     for (uint k = 0; k < kc; ++k) {
//       if (!recv)
//         v[offset++] = d_arrayCurr[index(i, j, k)];
//       else
//         d_arrayCurr[index(i, j, k)] = v[offset++];
//     }
//   }
// }

// void Equation::edgeY(ProcessorNode::Requests &requests, uint id, bool recv) {
//   ExchangeDir cdir = requests.iv[id];
//   thrust::host_vector<float> &v = requests.host[id];
//   uint offset = 0;
//   int j;
//   switch (cdir) {
//   case plus_y: {
//     j = recv ? jc : jc - 1;
//     break;
//   }
//   case minus_y: {
//     j = recv ? -1 : 0;
//     break;
//   }
//   }

//   for (uint i = 0; i < ic; ++i) {
//     for (uint k = 0; k < kc; ++k) {
//       inRange(offset, 0, v.size());
//       inRange(index(i, j, k), 0, d_arrayCurr.size());
//       if (!recv)
//         v[offset++] =d_arrayCurr[index(i, j, k)];
//       // copy_send(v, arrayCurr, i, j, k, offset++);
//       else
//         d_arrayCurr[index(i, j, k)] = v[offset++];
//       // copy_recv(v, arrayCurr, i, j, k, offset++);
//     }
//   }
// }

// void Equation::edgeZ(ProcessorNode::Requests &requests, uint id, bool recv) {
//   ExchangeDir cdir = requests.iv[id];
//   std::vector<float> &v = requests.host[id];
//   uint offset = 0;
//   int k;
//   switch (cdir) {
//   case plus_z: {
//     k = recv ? kc : kc - 1;
//     break;
//   }
//   case minus_z: {
//     k = recv ? -1 : 0;
//     break;
//   }
//   case period_plus_z: {
//     k = recv ? kc : kc - 1;
//     break;
//   }
//   case period_minus_z: {
//     k = recv ? 0 : 1;
//     break;
//   }
//   }
//   std::vector<float> &a =
//       (((cdir == period_minus_z) && recv) ? arrayNext : arrayCurr);
//   for (uint i = 0; i < ic; ++i) {
//     for (uint j = 0; j < jc; ++j) {
//       inRange(offset, 0, v.size());
//       inRange(index(i, j, k), 0, arrayCurr.size());
//       if (!recv)
//         v[offset++] = arrayCurr[index(i, j, k)];
//       else
//         arrayCurr[index(i, j, k)] = v[offset++];
//     }
//   }
// }

void Equation::copy(ProcessorNode::Requests &requests, uint id, bool recv) {
  inRange(id, 0, requests.iv.size());

  ExchangeDir cdir = requests.iv[id];
  switch (cdir) {
  case plus_x:
  case minus_x:
  case period_plus_x:
  case period_minus_x: {
    cuda_edgeX(requests.iv[id], requests.device[id], id, recv, d_arrayNext, d_arrayCurr);
    break;
  }
  case plus_y:
  case minus_y: {
    cuda_edgeY(requests.iv[id], requests.device[id], id, recv, d_arrayNext, d_arrayCurr);
    break;
  }
  case plus_z:
  case minus_z:
  case period_plus_z:
  case period_minus_z: {
    cuda_edgeZ(requests.iv[id], requests.device[id], id, recv, d_arrayNext, d_arrayCurr);
    break;
  }
  default:
    assert(false);
  }
}


// void Equation::calculateDir(ExchangeDir cdir) {
//   switch (cdir) {
//   case plus_x:
//   case minus_x: {
//     uint i = (cdir == plus_x) ? ic - 1 : 0;
//     for (uint j = 1; j < jc - 1; ++j) {
//       for (uint k = 1; k < kc - 1; ++k)
//         calculateIndex(i, j, k);
//     }
//     break;
//   }
//   case plus_y:
//   case minus_y: {
//     uint j = (cdir == plus_y) ? jc - 1 : 0;
//     for (uint i = 1; i < ic - 1; ++i) {
//       for (uint k = 1; k < kc - 1; ++k)
//         calculateIndex(i, j, k);
//     }
//     break;
//   }
//   case plus_z:
//   case minus_z: {
//     uint k = (cdir == plus_z) ? kc - 1 : 0;
//     for (uint i = 1; i < ic - 1; ++i) {
//       for (uint j = 1; j < jc - 1; ++j)
//         calculateIndex(i, j, k);
//     }
//     break;
//   }
//   case period_plus_x:
//   case period_minus_x:
//   case period_plus_z:
//   case period_minus_z:
//     break;
//   default:
//     assert(false);
//   }
// }

void Equation::inRange(int i, int a, int b) 
{ 
  assert((i >= a) && (i < b)); 
}


void ProcessorNode::init() {
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  size = world_size;

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  rank = world_rank;

  PhysSize = getPhysGrid(size);
  z = rank % PhysSize.z;
  y = rank / PhysSize.z % PhysSize.y;
  x = rank / PhysSize.z / PhysSize.y;
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

bool ProcessorNode::is(ExchangeDir cdir) const { return neighbor(cdir) != -1; }

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
  // buffs.push_back(std::vector<double>());
  // buffs.back().resize(sz);
}

//
