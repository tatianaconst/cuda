#include "cuda_equation.hpp"

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  node.init();
  print(node.rank, "Processor_Init OK\n");
  double t_free_start, t_free_finish;
  double start, finish;

  for (uint N = 128; N <= 512; N = N * 2) {
    double res;
    {     
      start = MPI_Wtime();

      r.clear();
      r.cpu = node.size;
      r.N = N;
      Equation eq(N);  
      eq.init();
      res = eq.run();
      t_free_start = MPI_Wtime();
    }
    t_free_finish = MPI_Wtime();
    r.t_free += t_free_finish - t_free_start;

    MPI_Barrier(MPI_COMM_WORLD);
    finish = MPI_Wtime();
    r.t_full = finish - start;

    double sum_host, sum_MPI, sum_calc, sum_init, sum_free, sum_full;

    MPI_Reduce(&r.t_host_device, &sum_host, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&r.t_MPI, &sum_MPI, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&r.t_calculation, &sum_calc, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&r.t_init, &sum_init, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&r.t_free, &sum_free, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&r.t_full, &sum_full, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    if (node.rank == 0)
      std::cout  << "###" << r.cpu 
                 << "," << r.N 
                 << "," << sum_host / r.cpu 
                 << "," << sum_MPI / r.cpu 
                 << "," << sum_calc / r.cpu 
                 << "," << sum_init / r.cpu 
                 << "," << sum_free / r.cpu 
                 << "," << sum_full / r.cpu 
                 << std::endl
                 << " " << res << std::endl;
    }
  MPI_Finalize();
  return 0;
}
