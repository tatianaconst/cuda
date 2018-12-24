#ifndef PLAIN_HPP
#define PLAIN_HPP

#include <vector>
#include "params.hpp"

class Equation {
 public:
  long N;
  int curr_step;
  uint i0, j0, k0;

  long ic, jc, kc;
  long incsize;

  double hx, hy, hz;

  uint ht;  // delta t

  rvector arrayPrev, arrayCurr, arrayNext;

  double sumResidual;

  double deltaTime(uint n) const { return n * ht; }

  Equation(uint n);

  void init();
  void run();

  void inRange(int i, int a, int b);

  void copy(ProcessorNode::Requests& requests, uint id, bool recv);

  void edgeX(ProcessorNode::Requests& requests, uint id, bool recv);
  void edgeY(ProcessorNode::Requests& requests, uint id, bool recv);
  void edgeZ(ProcessorNode::Requests& requests, uint id, bool recv);
  ProcessorNode::Requests recv_requests;
  ProcessorNode::Requests send_requests;

  void calculateIndex(uint i, uint j, uint k);
  void calculateDir(ExchangeDir cdir);

  uint index(uint i, uint j, uint k) const;

  double x(uint i) const;
  double y(uint j) const;
  double z(uint k) const;

  void initPrev();
  void initCurr();
};

#endif
