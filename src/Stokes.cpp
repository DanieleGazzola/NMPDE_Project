 #include "Stokes.hpp"

void Stokes::setup(){
    pcout << "Initializing the mesh" << std::endl;

    Triangulation<dim> mesh_serial;

    GridIn<dim> grid_in;
    grid_in.attach_triangulation(mesh_serial);

    const std::string mesh_file_name = "../mesh/mesh.msh";

    std::ifstream grid_in_file(mesh_file_name);
    grid_in.read_msh(grid_in_file);

    GridTools::partition_triangulation(mpi_size, mesh_serial);
    const auto construction_data = TriangulationDescription::Utilities::create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
    mesh.create_triangulation(construction_data);

    pcout << "  Number of elements = " << mesh.n_global_active_cells() << std::endl;
    pcout << "Initializing the finite element space" << std::endl;

    const FE_SimplexP<dim> fe_scalar_velocity(degree_velocity);
    const FE_SimplexP<dim> fe_scalar_pressure(degree_pressure);
    fe = std::make_unique<FESystem<dim>>(fe_scalar_velocity, dim, fe_scalar_pressure, 1);

    pcout << "  Velocity degree:           = " << fe_scalar_velocity.degree << std::endl;
    pcout << "  Pressure degree:           = " << fe_scalar_pressure.degree << std::endl;
    pcout << "  DoFs per cell              = " << fe->dofs_per_cell << std::endl;

    quadrature = std::make_unique<QGaussSimplex<dim>>(fe->degree + 1);

    pcout << "  Quadrature points per cell = " << quadrature->size() << std::endl;
    pcout << "-----------------------------------------------" << std::endl;

    quadrature_boundary = std::make_unique<QGaussSimplex<dim - 1>>(fe-> degree + 1);

    std::cout << "  Quadrature points per boundary cell = "
              << quadrature_boundary->size() << std::endl;

    pcout << "Initializing the DoF handler" << std::endl;

    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    // We want to reorder DoFs so that all velocity DoFs come first, and then
    // all pressure DoFs.
    std::vector<unsigned int> block_component(dim + 1, 0);
    block_component[dim] = 1;
    DoFRenumbering::component_wise(dof_handler, block_component);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    // Besides the locally owned and locally relevant indices for the whole
    // system (velocity and pressure), we will also need those for the
    // individual velocity and pressure blocks.
    std::vector<types::global_dof_index> dofs_per_block =
      DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
    const unsigned int n_u = dofs_per_block[0];
    const unsigned int n_p = dofs_per_block[1];

    block_owned_dofs.resize(2);
    block_relevant_dofs.resize(2);
    block_owned_dofs[0]    = locally_owned_dofs.get_view(0, n_u);
    block_owned_dofs[1]    = locally_owned_dofs.get_view(n_u, n_u + n_p); 
    block_relevant_dofs[0] = locally_relevant_dofs.get_view(0, n_u);
    block_relevant_dofs[1] = locally_relevant_dofs.get_view(n_u, n_u + n_p);

    pcout << "  Number of DoFs: " << std::endl;
    pcout << "    velocity = " << n_u << std::endl;
    pcout << "    pressure = " << n_p << std::endl;
    pcout << "    total    = " << n_u + n_p << std::endl;

    pcout << "Initializing the linear system" << std::endl;
    pcout << "  Initializing the sparsity pattern" << std::endl;

    Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);
    for (unsigned int c = 0; c < dim + 1; ++c){
        for (unsigned int d = 0; d < dim + 1; ++d){
            if (c == dim && d == dim)
                coupling[c][d] = DoFTools::none;
            else
                coupling[c][d] = DoFTools::always;
        }
    }

    TrilinosWrappers::BlockSparsityPattern sparsity(block_owned_dofs, MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, coupling, sparsity);
    sparsity.compress();
    
    for (unsigned int c = 0; c < dim + 1; ++c){
        for (unsigned int d = 0; d < dim + 1; ++d){
            if (c == dim && d == dim)
                coupling[c][d] = DoFTools::always;
            else
                coupling[c][d] = DoFTools::none;
        }
    }
    TrilinosWrappers::BlockSparsityPattern sparsity_pressure_mass(block_owned_dofs, MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, coupling, sparsity_pressure_mass);
    sparsity_pressure_mass.compress();

    for (unsigned int c = 0; c < dim + 1; ++c){
        for (unsigned int d = 0; d < dim + 1; ++d){
            if (c == dim || d == dim)
                coupling[c][d] = DoFTools::none;
            else
                coupling[c][d] = DoFTools::always;
        }
    }
    TrilinosWrappers::BlockSparsityPattern sparsity_rhs_matrix(block_owned_dofs, MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, coupling, sparsity_rhs_matrix);
    sparsity_rhs_matrix.compress();

    pcout << "  Initializing the matrices" << std::endl;
    system_matrix.reinit(sparsity);
    pressure_mass.reinit(sparsity_pressure_mass);
    rhs_matrix.reinit(sparsity_rhs_matrix);

    pcout << "  Initializing the system right-hand side" << std::endl;
    system_rhs.reinit(block_owned_dofs, MPI_COMM_WORLD);
    pcout << "  Initializing the solution vector" << std::endl;
    solution_owned.reinit(block_owned_dofs, MPI_COMM_WORLD);
    solution.reinit(block_owned_dofs, block_relevant_dofs, MPI_COMM_WORLD);

    force_file.open("force.csv", std::ofstream::out | std::ofstream::app);
}

void Stokes::assembleMatrices(){
    pcout << "===============================================" << std::endl;
    pcout << "Assembling matrices of the system" << std::endl;

    const unsigned int dofs_per_cell = fe->dofs_per_cell;
    const unsigned int n_q = quadrature->size();

    FEValues<dim> fe_values(*fe, *quadrature, update_values | update_gradients | update_quadrature_points | update_JxW_values);

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_pressure_mass_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_rhs_matrix(dofs_per_cell);

    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    system_matrix = 0.0;
    pressure_mass = 0.0;
    rhs_matrix = 0.0;

    FEValuesExtractors::Vector velocity(0);
    FEValuesExtractors::Scalar pressure(dim);

    std::vector<Tensor<1, dim>> solution_loc(n_q);

    for (const auto &cell : dof_handler.active_cell_iterators()){
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);

        cell_matrix = 0.0;
        cell_pressure_mass_matrix = 0.0;
        cell_rhs_matrix = 0.0;

        fe_values[velocity].get_function_values(solution.block(0), solution_loc);

        for(unsigned int q = 0; q < n_q; ++q){
            for(unsigned int i = 0; i < dofs_per_cell; ++i){
                for(unsigned int j = 0; j < dofs_per_cell; ++j){
                    cell_matrix(i, j) += 1/Reynolds * scalar_product(fe_values[velocity].gradient(i, q), fe_values[velocity].gradient(j, q)) * fe_values.JxW(q);
                    cell_matrix(i, j) += scalar_product(fe_values[velocity].value(i, q),fe_values[velocity].value(j,q)) / deltaT * fe_values.JxW(q);
                    cell_matrix(i, j) += scalar_product(fe_values[velocity].gradient(i, q) * solution_loc[q], fe_values[velocity].value(j,q)) * fe_values.JxW(q);

                    cell_rhs_matrix(i, j) += scalar_product(fe_values[velocity].value(i, q), fe_values[velocity].value(j,q)) / deltaT * fe_values.JxW(q);

                    cell_matrix(i, j) -= fe_values[velocity].divergence(i, q) * fe_values[pressure].value(j, q) * fe_values.JxW(q);
                    cell_matrix(i, j) -= fe_values[velocity].divergence(j, q) * fe_values[pressure].value(i, q) * fe_values.JxW(q);

                    cell_pressure_mass_matrix(i, j) += fe_values[pressure].value(i, q) * fe_values[pressure].value(j, q) * Reynolds * fe_values.JxW(q);
                }
            }
        }

        cell->get_dof_indices(dof_indices);

        system_matrix.add(dof_indices, cell_matrix);
        pressure_mass.add(dof_indices, cell_pressure_mass_matrix);
        rhs_matrix.add(dof_indices, cell_rhs_matrix);
    }

    system_matrix.compress(VectorOperation::add);
    pressure_mass.compress(VectorOperation::add);
    rhs_matrix.compress(VectorOperation::add);
}

void Stokes::solve()
{
    pcout << "===============================================" << std::endl;
    pcout << "Applying the initial condition" << std::endl;
    //const unsigned int mpi_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

    VectorTools::interpolate(dof_handler, u_0, solution_owned);
    solution = solution_owned;

    output(0, 0.0);
    pcout << "-----------------------------------------------" << std::endl;

    unsigned int time_step = 0;
    double time = 0;

    while (time < T){
        time += deltaT;
        ++time_step;

        pcout << "n = " << std::setw(3) << time_step << ", t = " << std::setw(5) << time << ":" << std::flush;

        assembleMatrices();
        assembleRhs(time);
        solveTimeStep();
        output(time_step, time);
        calculateForce();
    }
}

void Stokes::assembleRhs(const double &time){
    pcout << "Assembling right-hand side of the system" << std::endl;

    const unsigned int dofs_per_cell = fe->dofs_per_cell;
    const unsigned int n_q = quadrature->size();

    FEValues<dim> fe_values(*fe, *quadrature, update_values | update_gradients | update_quadrature_points | update_JxW_values);

    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    system_rhs = 0.0;

    FEValuesExtractors::Vector velocity(0);
    FEValuesExtractors::Scalar pressure(dim);

    for (const auto &cell : dof_handler.active_cell_iterators()){
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);

        cell_rhs = 0.0;

        for(unsigned int q = 0; q < n_q; ++q){
            forcing_term.set_time(time);
            Vector<double> forcing_term_loc(dim);
            forcing_term.vector_value(fe_values.quadrature_point(q), forcing_term_loc);
            Tensor<1, dim> forcing_term_tensor;

            for (unsigned int d = 0; d < dim; ++d){
                forcing_term_tensor[d] = forcing_term_loc[d];
            }

            for(unsigned int i = 0; i < dofs_per_cell; ++i){
                cell_rhs(i) += scalar_product(forcing_term_tensor, fe_values[velocity].value(i, q)) * fe_values.JxW(q);
            }
        }
        cell->get_dof_indices(dof_indices);

        system_rhs.add(dof_indices, cell_rhs);
    }

    rhs_matrix.vmult_add(system_rhs, solution_owned);

    std::map<types::global_dof_index, double> boundary_values;
    std::map<types::boundary_id, const Function<dim> *> boundary_functions;

    Functions::ZeroFunction<dim> zero_function(dim + 1);

    inlet_velocity.set_time(time);

    boundary_functions[0] = &inlet_velocity;
    boundary_functions[1] = &zero_function;
    boundary_functions[3] = &zero_function;
    boundary_functions[4] = &zero_function;
    boundary_functions[5] = &zero_function;
    boundary_functions[6] = &zero_function;
    boundary_functions[7] = &zero_function;

    VectorTools::interpolate_boundary_values(dof_handler, boundary_functions, boundary_values, ComponentMask({true, true, false}));

    MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs, false);

    system_rhs.compress(VectorOperation::add);
    system_matrix.compress(VectorOperation::add);
    pressure_mass.compress(VectorOperation::add);
    rhs_matrix.compress(VectorOperation::add);
}

void Stokes::solveTimeStep(){
    pcout << "===============================================" << std::endl;

    SolverControl solver_control(100000, 1e-6 * system_rhs.l2_norm());

    SolverGMRES<TrilinosWrappers::MPI::BlockVector> solver(solver_control);

    PreconditionBlockTriangular preconditioner;
    preconditioner.initialize(system_matrix.block(0, 0), pressure_mass.block(1, 1), system_matrix.block(1, 0));

    pcout << "Solving the linear system" << std::endl;
    solver.solve(system_matrix, solution_owned, system_rhs, preconditioner);
    pcout << "  " << solver_control.last_step() << " GMRES iterations" << std::endl;

    solution = solution_owned;
}

void Stokes::output(const unsigned int &time_step, const double &time){
    pcout << "===============================================" << std::endl;

    DataOut<dim> data_out;

    std::vector<DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    std::vector<std::string> names = {"velocity", "velocity", "pressure"};

    data_out.add_data_vector(dof_handler, solution, names, data_component_interpretation);

    std::vector<unsigned int> partition_int(mesh.n_active_cells());
    GridTools::get_subdomain_association(mesh, partition_int);
    const Vector<double> partitioning(partition_int.begin(), partition_int.end());
    data_out.add_data_vector(partitioning, "partitioning");

    data_out.build_patches();

    std::string output_file_name = std::to_string(time_step);
    output_file_name = "output-" + std::string(4 - output_file_name.size(), '0') + output_file_name;

    DataOutBase::DataOutFilter data_filter( DataOutBase::DataOutFilterFlags(/*filter_duplicate_vertices = */ false, /*xdmf_hdf5_output = */ true));
    data_out.write_filtered_data(data_filter);
    data_out.write_hdf5_parallel(data_filter, output_file_name + ".h5", MPI_COMM_WORLD);

    std::vector<XDMFEntry> xdmf_entries({data_out.create_xdmf_entry(data_filter, output_file_name + ".h5", time, MPI_COMM_WORLD)});
    data_out.write_xdmf_file(xdmf_entries, output_file_name + ".xdmf", MPI_COMM_WORLD);

    pcout << "Output written to " << output_file_name << std::endl;
    pcout << "===============================================" << std::endl;
}

void Stokes::calculateForce(){
    FEFaceValues<dim> fe_values_boundary(*fe, *quadrature_boundary, update_values | update_gradients | update_quadrature_points | update_normal_vectors | update_JxW_values);
    
    FEValuesExtractors::Vector velocity(0);
    FEValuesExtractors::Scalar pressure(dim);
    
    const unsigned int n_q = quadrature_boundary->size();

    std::vector<Tensor<dim, dim>> velocity_gradient_loc(n_q);
    std::vector<double> pressure_loc(n_q);

    Tensor<1, dim> force;

    for (const auto &cell : dof_handler.active_cell_iterators()){
        if (!cell->is_locally_owned())
            continue;
        if (cell->at_boundary()){            
            for (unsigned int face_number = 0; face_number < cell->n_faces(); ++face_number){
                if (cell->face(face_number)->at_boundary() && (cell->face(face_number)->boundary_id() == 4 || cell->face(face_number)->boundary_id() == 5 || cell->face(face_number)->boundary_id() == 6 || cell->face(face_number)->boundary_id() == 7)){
                    fe_values_boundary.reinit(cell, face_number);
                    fe_values_boundary[velocity].get_function_gradients(solution.block(0), velocity_gradient_loc);
                    fe_values_boundary[pressure].get_function_values(solution.block(1), pressure_loc);
                    for (unsigned int q = 0; q < n_q; ++q)
                        force += (ro * nu * velocity_gradient_loc[q] * fe_values_boundary.normal_vector(q) - pressure_loc[q] * fe_values_boundary.normal_vector(q)) * fe_values_boundary.JxW(q);
                }
            }
        }
    }

    Vector<double> v(dim);
    for(unsigned int i = 0; i < dim; i++)
        v[i] = force[i];

    Utilities::MPI::sum(v, MPI_COMM_WORLD, v);
    const unsigned int mpi_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    if(mpi_rank == 0){
        for(unsigned int i = 0; i < dim; i++){
            force_file << v[i] << ";";
        }
        force_file << std::endl; 
    }
}