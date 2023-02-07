#include <deal.II/base/convergence_table.h>

#include <fstream>
#include <iostream>
#include <vector>

#include "Poisson3D_parallel.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const unsigned int N = 39;
  const unsigned int degree = 1;

  Poisson3DParallel problem(N, degree);

  problem.generateSphere(3, 0.18);
  problem.setup();
  problem.assemble();
  problem.solve();
  problem.output();

  return 0;
}