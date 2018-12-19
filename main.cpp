#include "cuda_equation.hpp"

int main(int argc, char **argv) {

  MPI_Init(&argc, &argv);
  node.init();
  print(node.rank, "Processor_Init OK\n");

  for (uint N = 128; N <= 1024; N = N * 2) {
    double start, finish;
    start = MPI_Wtime();
    Equation eq(N); // Equation parameters, send/recv buffers
    eq.init();
    eq.run();

    MPI_Barrier(MPI_COMM_WORLD);

    finish = MPI_Wtime();

    double resTime = finish - start;
    if (node.rank == 0)
      printf("Time: %lf\n", resTime);
  }
  MPI_Finalize();
  return 0;
}
