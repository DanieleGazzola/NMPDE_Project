#ifndef STOKES_HPP
#define STOKES_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_in.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

using namespace dealii;

class Stokes{
public:
    static constexpr unsigned int dim = 2;

    class ForcingTerm : public Function<dim>{
    public:
        virtual void vector_value(const Point<dim> & /*p*/, Vector<double> &values) const override{
            for(unsigned int i = 0; i < dim; ++i)
                values[i] = 0.;
        }

        virtual double value(const Point<dim> & /*p*/, const unsigned int /*component*/= 0) const override{
            return 0.;
        }
    };

    class InletVelocity : public Function<dim>{
    public:
        InletVelocity():
        Function<dim>(dim + 1)
        {}

        virtual void vector_value(const Point<dim> & p, Vector<double> &values) const override{
            values[0] = std::sin(M_PI * get_time() / 1) * 6 * p[1] * (0.41 - p[1])/(0.41*0.41);

            for (unsigned int i = 1; i < dim + 1; ++i)
                values[i] = 0.0;
        }

        virtual double value(const Point<dim> & p, const unsigned int component = 0) const override{
            if (component == 0)
                return std::sin(M_PI * get_time() / 8) * 6 * p[1] * (0.41 - p[1])/(0.41*0.41);
            else
                return 0.0;
        }
    };

    class FunctionU0 : public Function<dim>{
    public:
        FunctionU0():
        Function<dim>(dim + 1)
        {}

        virtual void vector_value(const Point<dim> & /*p*/, Vector<double> &values) const override{
            values[0] = 0.0;
            for (unsigned int i = 1; i < dim + 1; ++i)
                values[i] = 0.0;
        }

        virtual double value(const Point<dim> & /*p*/, const unsigned int component = 0) const override{
            if (component == 0)
                return 0.0;
            else return 0.0;
        }
    };

    class PreconditionBlockTriangular{
    public:
        void initialize(const TrilinosWrappers::SparseMatrix &velocity_stiffness_, const TrilinosWrappers::SparseMatrix &pressure_mass_, const TrilinosWrappers::SparseMatrix &B_){
            velocity_stiffness = &velocity_stiffness_;
            pressure_mass = &pressure_mass_;
            B = &B_;

            preconditioner_velocity.initialize(velocity_stiffness_);
            preconditioner_pressure.initialize(pressure_mass_);
        }

        void vmult(TrilinosWrappers::MPI::BlockVector & dst, const TrilinosWrappers::MPI::BlockVector &src) const{
            SolverControl solver_control_velocity(10000, 1e-2 * src.block(0).l2_norm());
            SolverGMRES<TrilinosWrappers::MPI::Vector> solver_velocity(solver_control_velocity);
            solver_velocity.solve(*velocity_stiffness, dst.block(0), src.block(0), preconditioner_velocity);

            tmp.reinit(src.block(1));
            B->vmult(tmp, dst.block(0));
            tmp.sadd(-1., src.block(1));

            SolverControl solver_control_pressure(10000, 1e-2 * src.block(1).l2_norm());
            SolverGMRES<TrilinosWrappers::MPI::Vector> solver_pressure(solver_control_pressure);
            solver_velocity.solve(*pressure_mass, dst.block(1), tmp, preconditioner_pressure);
        } 

    protected:
        const TrilinosWrappers::SparseMatrix *velocity_stiffness;

        TrilinosWrappers::PreconditionILU preconditioner_velocity;

        const TrilinosWrappers::SparseMatrix *pressure_mass;

        TrilinosWrappers::PreconditionILU preconditioner_pressure;

        const TrilinosWrappers::SparseMatrix *B;

        mutable TrilinosWrappers::MPI::Vector tmp;   
    };

    Stokes(const unsigned int &N_, const unsigned int &degree_velocity_, const unsigned int &degree_pressure_, const double &T_, const double &deltaT_) :
        mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)),
        mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)),
        pcout(std::cout, mpi_rank == 0),
        N(N_),
        T(T_),
        deltaT(deltaT_),
        degree_velocity(degree_velocity_),
        degree_pressure(degree_pressure_),
        mesh(MPI_COMM_WORLD)
        {}

    void setup();
    void assembleMatrices();
    void solve();
    void assembleRhs(const double &time);
    void solveTimeStep();
    void output(const unsigned int &time_step, const double &time);
    void calculateForce();

protected:
    const unsigned int mpi_size;
    const unsigned int mpi_rank;

    ConditionalOStream pcout;
    
    const double Reynolds = 10;
    const double ro = 1;
    const double nu = 1e-3;
    ForcingTerm forcing_term;
    InletVelocity inlet_velocity;
    FunctionU0 u_0;
  
    const unsigned int N; 
    const double T;   //final time
    const double deltaT;

    const unsigned int degree_velocity;
    const unsigned int degree_pressure;

    parallel::fullydistributed::Triangulation<dim> mesh;

    std::unique_ptr<FiniteElement<dim>> fe;

    std::unique_ptr<Quadrature<dim>> quadrature;
    std::unique_ptr<Quadrature<dim - 1>> quadrature_boundary;

    DoFHandler<dim> dof_handler;
    IndexSet locally_owned_dofs;    
    std::vector<IndexSet> block_owned_dofs;
    IndexSet locally_relevant_dofs;

    std::vector<IndexSet> block_relevant_dofs;

    TrilinosWrappers::BlockSparseMatrix system_matrix;
    TrilinosWrappers::BlockSparseMatrix pressure_mass;
    TrilinosWrappers::BlockSparseMatrix rhs_matrix;
    TrilinosWrappers::MPI::BlockVector system_rhs;
    TrilinosWrappers::MPI::BlockVector solution_owned;
    TrilinosWrappers::MPI::BlockVector solution;

    std::ofstream force_file;
};

#endif