#ifndef EQUATION_HPP
#define EQUATION_HPP

#include "cuda_param.hpp"
#include <vector>


struct Reporter
{
  Reporter()
  :cpu(0), N(0), t_host_device(0), t_MPI(0),
   t_calculation(0), t_init(0), t_free(0),
   t_offset(0), t_full(0)
  {}

  int cpu;
  int N;
  double t_host_device;
  double t_MPI;
  double t_calculation;
  double t_init;
  double t_free;
  double t_offset;
  double t_full;
};

extern Reporter r;

struct ProcessorNode {
  uint rank;
  uint size;

  PhysSize PhysSize;

  int x;
  int y;
  int z;

  void init();

  int neighbor(ExchangeDir cdir) const;
  bool is(ExchangeDir cdir) const;

  int toRank(uint i, uint j, uint k) const;

  struct Requests {

    ExchangeDir dir;

    std::vector<MPI_Request> v;
    std::vector<hvector> host;
    std::vector<dvector> device;
    std::vector<ExchangeDir> iv;

    uint size() const { return v.size(); }
    void append(ExchangeDir dir, uint sz);
    void gpu_to_cpu()
    {
      for (int i = 0; i < size(); ++i){
        host[i] = device[i];
      }
    }
    void cpu_to_gpu()
    {
      for (int i = 0; i < size(); ++i){
        device[i] = host[i];
      }
    }

  };
};

class Equation {
public:
  long N;
  int curr_step;
  uint i0, j0, k0;

  long ic, jc, kc;
  long incsize;

  double hx, hy, hz;

  uint ht; // delta t
  int K;
  double T;

  dvector d_arrayPrev, d_arrayCurr, d_arrayNext;

  Equation(uint n);

  void init();
  void run();

  void inRange(int i, int a, int b);

  void copy(ProcessorNode::Requests &requests, uint id, bool recv);

  ProcessorNode::Requests recv_requests;
  ProcessorNode::Requests send_requests;

  void calculateIndex(uint i, uint j, uint k);
  void calculateDir(ExchangeDir cdir);

};

extern ProcessorNode node;

#endif // Equation_HPP
