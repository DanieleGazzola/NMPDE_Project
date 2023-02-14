#include "Stokes.hpp"

// Main function.
int
main(int argc, char *argv[])
{
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

    const unsigned int N = 19;
    const unsigned int degree_velocity = 2;
    const unsigned int degree_pressure = 1;

    const double T = 3.2;
    const double deltat = 0.01;

    Stokes problem(N, degree_velocity, degree_pressure, T, deltat);

    problem.setup();
    problem.solve();

    return 0;
}