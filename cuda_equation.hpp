#ifndef EQUATION_HPP
#define EQUATION_HPP

#include "cuda_param.hpp"
#include <vector>

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
    //std::vector<rvector> buffs;
    std::vector<hvector> host;
    std::vector<dvector> device;
    std::vector<ExchangeDir> iv;

    uint size() const { return v.size(); }
    void append(ExchangeDir dir, uint sz);
    void gpu_to_cpu()
    {
      for (int i = 0; i < size(); ++i)
        host[i] = device[i];
    }
    void cpu_to_gpu()
    {
      for (int i = 0; i < size(); ++i)
        device[i] = host[i];
    }

    //Requests() { buffs.reserve(10); }
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

  dvector d_arrayPrev, d_arrayCurr, d_arrayNext;

  // rvector 

  // double maxResidual;

  // double deltaTime(uint n) const { return n * ht; }

  Equation(uint n);

  void init();
  void run();

  void inRange(int i, int a, int b);

  void copy(ProcessorNode::Requests &requests, uint id, bool recv);

  // void edgeX(ProcessorNode::Requests &requests, uint id, bool recv);
  // void edgeY(ProcessorNode::Requests &requests, uint id, bool recv);
  // void edgeZ(ProcessorNode::Requests &requests, uint id, bool recv);
  ProcessorNode::Requests recv_requests;
  ProcessorNode::Requests send_requests;

  void calculateIndex(uint i, uint j, uint k);
  void calculateDir(ExchangeDir cdir);

  // uint index(uint i, uint j, uint k) const;

  // double x(uint i) const;
  // double y(uint j) const;
  // double z(uint k) const;

  // void initPrev();
  // void initCurr();
};

extern ProcessorNode node;

#endif // Equation_HPP
