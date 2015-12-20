#include "shifted_matrix.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>

#include "eigen_inverse_m.h"

#include <fstream>
#include <sstream>
#include <iostream>

#define dim 2

using namespace dealii;

/*************************************************************************/
/*************************************************************************/
class PolarSolver {
  public:
    PolarSolver(const double Lx, const double Ly, const double R);
    ~PolarSolver();

  // Make the initial grid and initialize the texture to zero.
  void make_initial_grid();

  // Refine the grid according with the texture gradients
  // and redistribute the texture in the new grid.
  double refine_grid(const double ref, const double crs);

  // save the grid and a data vector in a eps file
  void save_grid(const char* fname);
  void save_data(const char* fname, const Vector<double> & data);

  // Do the texture calculations.
  // repeat=true shows that texture didn't change after the last calculation
  void do_text_calc(bool repeat=false);

  // calculate texture accuracy by integrating (nabla^2 a - sin(2a)/2)^2
  double check_text();

  // Do the wave calculation.
  void do_wave_calc();

  // Setup boundary IDs. Used in make_initial_grid() and refine_grid().
  void setup_boundary_ids();

  Vector<double>       texture;
  Vector<double>       wave;
  double               en;

  private:
    Triangulation<dim>     triang;
    DoFHandler<dim>        dofs;
    FE_Q<dim>              fe;
    ConstraintMatrix     constraints;
    SparsityPattern      sparsity;
    double Lx, Ly, R;


    SparseMatrix<double> A,M;
    Vector<double>       B;

    bool wave_zero_bc;
    bool wave_nopot;
};

PolarSolver::PolarSolver(
     const double Lx_, const double Ly_, const double R_):
     dofs(triang), fe(2), Lx(Lx_), Ly(Ly_), R(R_) {
  wave_zero_bc = false;
  wave_nopot = false;
}

PolarSolver::~PolarSolver() {
  dofs.clear();
  triang.clear();
}

/*************************************************************************/
// reset boundary IDs:
// 1: x=0
// 2: y=0, x<R
// 3: other
void
PolarSolver::setup_boundary_ids(){
  typename Triangulation<dim>::cell_iterator
    cell = triang.begin(),
    endc = triang.end();
  for (; cell!=endc; ++cell){
    for (unsigned int i=0; i<GeometryInfo<dim>::faces_per_cell; i++){
      if (cell->face(i)->boundary_id() ==
          numbers::internal_face_boundary_id) continue;

      if (R<Lx){ // finite soliton
        if (std::fabs(cell->face(i)->center()(0)) < 1e-12)
          cell->face(i)->set_boundary_id(1);
        else if (std::fabs(cell->face(i)->center()(0)) < R &&
                 std::fabs(cell->face(i)->center()(1)) < 1e-12)
          cell->face(i)->set_boundary_id(2);
        else
          cell->face(i)->set_boundary_id(0);
      }
      else { // infinite soliton
        if (std::fabs(cell->face(i)->center()(0)) < 1e-12 ||
            std::fabs(cell->face(i)->center()(0) - Lx) < 1e-12)
          cell->face(i)->set_boundary_id(1);
        else if (std::fabs(cell->face(i)->center()(1)) < 1e-12)
          cell->face(i)->set_boundary_id(2);
        else
          cell->face(i)->set_boundary_id(0);
      }
    }
  }

}


// Make the initial grid and initialize the texture to zero.
void
PolarSolver::make_initial_grid(){
  // rectangular grid 1x1
  GridGenerator::hyper_rectangle(triang,
    Point<dim>(0,0), Point<dim>(Lx,Ly));

  // refining the grid until the interesting region 0:R
  // will be covered by more then one cell (and even more fine).
  while(1){
    bool do_refine=false;
    typename Triangulation<dim>::cell_iterator
      cell = triang.begin(),
      endc = triang.end();
    for (; cell!=endc; ++cell){
      // skip unwanted cells:
      if (!cell->at_boundary() ||
          !cell->used() || !cell->active()) continue;
      for (unsigned int i=0; i<GeometryInfo<dim>::faces_per_cell; i++){
        // skip internal faces
        if (cell->face(i)->boundary_id() ==
            numbers::internal_face_boundary_id) continue;
        // skip all faces save y=0
        if (std::fabs(cell->face(i)->center()(1)) > 1e-12) continue;
        double x1 = cell->face(i)->vertex(0)(0);
        double x2 = cell->face(i)->vertex(1)(0);
        if ((x1-R)*(x2-R)<0 && fabs(x2-x1)>R/16) {
          cell->set_refine_flag();
          do_refine=true;
        }
      }
    }
    if (!do_refine) break;
    triang.execute_coarsening_and_refinement();
  }

  // special case: infinite soliton
  if (R>=Lx) triang.refine_global(4);

  // enumerate degrees of freedom
  dofs.distribute_dofs(fe);

  // set correct boundary IDs
  setup_boundary_ids();

  // initialize initial texture
  texture.reinit(dofs.n_dofs());
}


// Refine the grid according with the texture gradients
// and redistribute the texture.
double
PolarSolver::refine_grid(const double ref, const double crs){
  Vector<float> err(triang.n_active_cells());

  // estimate error per cell
  KellyErrorEstimator<dim>::estimate(dofs, QGauss<dim-1>(3),
     typename FunctionMap<dim>::type(), texture, err);

  // mark cells for refinement(10% worse to refine, 3% best to coarse)
  GridRefinement::refine_and_coarsen_fixed_number(triang, err, ref, crs);

  // mark additional cells to be refined
  triang.prepare_coarsening_and_refinement();

  // prepare transfer of the solution to the new mesh
  SolutionTransfer<dim> solution_transfer(dofs);
  solution_transfer.prepare_for_coarsening_and_refinement(texture);

  // do actual rifinement
  triang.execute_coarsening_and_refinement();

  // enumerate degrees of freedom
  dofs.distribute_dofs(fe);

  // do transfer of the solution to the new mesh
  Vector<double> tmp(dofs.n_dofs());
  solution_transfer.interpolate(texture, tmp);
  texture = tmp;

  // set correct boundary IDs
  setup_boundary_ids();

  return err.mean_value();
}

/*************************************************************************/
// save the grid and a data vector in a eps file
void
PolarSolver::save_grid(const char* fname){
  GridOut grid_out;
  std::ofstream out(fname);
  grid_out.write_eps(triang,out);
}
void
PolarSolver::save_data(const char* fname, const Vector<double> & data){
  DataOutBase::EpsFlags eps_flags;

  eps_flags.z_scaling = 1./data.linfty_norm();
  eps_flags.draw_mesh = false;
  eps_flags.azimut_angle = 40;
  eps_flags.turn_angle   = 130;

  DataOut<dim> data_out;
  data_out.set_flags(eps_flags);

  data_out.attach_dof_handler(dofs);
  data_out.add_data_vector(data, "data");

  data_out.build_patches();

  std::ofstream out(fname);
  data_out.write_eps(out);
}


/*************************************************************************/
// Do the texture calculations.
// repeat=true shows that texture didn't change after the last calculation
void
PolarSolver::do_text_calc(bool repeat){

  if (!repeat){
    // set text constraints
    constraints.clear();
    DoFTools::make_hanging_node_constraints(dofs, constraints);
    VectorTools::interpolate_boundary_values(
        dofs, 0, ZeroFunction<dim>(), constraints);
    VectorTools::interpolate_boundary_values(
        dofs, 2, ConstantFunction<dim>(0.5*M_PI), constraints);
    constraints.close();

    // extend texture to constrained points (maybe not needed, but safe)
    constraints.distribute(texture);

    // rebuild sparsity pattern for the matrix
    DynamicSparsityPattern dsp(dofs.n_dofs());
    DoFTools::make_sparsity_pattern(dofs, dsp);
    constraints.condense(dsp);
    sparsity.copy_from(dsp);

  }

  // build matrices
  {
    A.reinit(sparsity);
    B.reinit(dofs.n_dofs());

    const QGauss<dim>  quadrature_formula(3);
    FEValues<dim> fe_values(fe, quadrature_formula,
                update_values | update_gradients |
                update_quadrature_points | update_JxW_values);

    const unsigned int   nd = fe.dofs_per_cell;
    const unsigned int   nq = quadrature_formula.size();

    // cell matrices
    FullMatrix<double>   Acell(nd,nd);
    Vector<double>       Bcell(nd);
    std::vector<types::global_dof_index> local_dof_indices(nd);

    // texture in the cell
    std::vector<double> texture_cell(nq);

    typename DoFHandler<dim>::active_cell_iterator
      cell = dofs.begin_active(),
      endc = dofs.end();
    for (; cell!=endc; ++cell) {
      Acell = 0;
      Bcell = 0;

      // update existing solution on the sell for iterative process
      fe_values.reinit(cell);
      fe_values.get_function_values(texture, texture_cell);

      // here I solve: nabla^2 Un+1 - Un+1 = sin(2Un)/2 - Un
      for (unsigned int q=0; q<nq; ++q){
        for (unsigned int i=0; i<nd; ++i){
          for (unsigned int j=0; j<nd; ++j){
            Acell(i,j) -=(fe_values.shape_grad(i,q) *
                          fe_values.shape_grad(j,q) *
                          fe_values.JxW(q));
            Acell(i,j) -=(fe_values.shape_value(i,q) *
                          fe_values.shape_value(j,q) *
                          fe_values.JxW(q));
          }
          Bcell(i) += (fe_values.shape_value(i,q) *
                       0.5*sin(2.0*texture_cell[q]) *
                       fe_values.JxW(q));
          Bcell(i) -= (fe_values.shape_value(i,q) *
                       texture_cell[q] *
                       fe_values.JxW(q));
        }
      }
      // Finally, transfer the contributions from Acell and
      // Bcell into the global objects.
      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(
         Acell, Bcell, local_dof_indices, A, B);
    }
    A.compress (VectorOperation::add);
  }


  // print some information about the cell and dof
  std::cout << "  texture calculation: "
            << "  act.cells: " << std::setw(4) << triang.n_active_cells()
            << "  DOFs: " << std::setw(4) << dofs.n_dofs();


  // solve the system
  {
    SolverControl      solver_control(1000, 1e-12);
    SolverCG<>         solver(solver_control);

    PreconditionSSOR<> preconditioner;
    preconditioner.initialize(A, 1.2);
    solver.solve(A, texture, B, preconditioner);
    constraints.distribute(texture);
  }

  std::cout << "  done" << std::endl;

}


/*************************************************************************/
// Check the texture accuracy.
double
PolarSolver::check_text(){

  const QGauss<dim>  quadrature_formula(3);
  FEValues<dim> fe_values(fe, quadrature_formula,
              update_values | update_gradients | update_hessians |
              update_quadrature_points | update_JxW_values);

  const unsigned int   nq = quadrature_formula.size();

  // texture in the cell's quadrature points
  // same for laplacian
  std::vector<double> cell_val(nq);
  std::vector<double> cell_lap(nq);
  std::vector<Tensor<1,dim> > cell_grad(nq);

  double I1=0; // integral
  typename DoFHandler<dim>::active_cell_iterator cell;
  for (cell= dofs.begin_active(); cell!=dofs.end(); ++cell) {
    fe_values.reinit(cell);
    fe_values.get_function_values(texture, cell_val);
    fe_values.get_function_laplacians(texture, cell_lap);
    fe_values.get_function_gradients(texture, cell_grad);

    for (unsigned int q=0; q<nq; ++q){

      std::cerr << "> " << cell_val[q]
                << " "  << cell_lap[q]
                << " "  << cell_grad[q].norm_square()
                << "\n";

      double v = cell_lap[q] - 0.5*sin(2.0*cell_val[q]);
      I1 += v*v * fe_values.JxW(q);
    }
    std::cerr << "\n";
  }
  return sqrt(I1);
}


/*************************************************************************/
// do the wave calculations
void
PolarSolver::do_wave_calc(){

  // set wave constraints
  constraints.clear();
  DoFTools::make_hanging_node_constraints(dofs, constraints);
  VectorTools::interpolate_boundary_values(
      dofs, 0, ZeroFunction<dim>(), constraints);
  if (wave_zero_bc){ // zero BC everywhere - for test
    VectorTools::interpolate_boundary_values(
        dofs, 1, ZeroFunction<dim>(), constraints);
    VectorTools::interpolate_boundary_values(
        dofs, 2, ZeroFunction<dim>(), constraints);
  }
  constraints.close();

  // init the wave vector
  wave.reinit(dofs.n_dofs());

  // rebuild sparsity pattern for the matrix
  DynamicSparsityPattern dsp(dofs.n_dofs());
  DoFTools::make_sparsity_pattern(dofs, dsp);
  constraints.condense(dsp);
  sparsity.copy_from(dsp);

  // build matrices
  {
    A.reinit(sparsity);
    M.reinit(sparsity);
    B.reinit(dofs.n_dofs());

    const QGauss<dim>  quadrature_formula(3);
    FEValues<dim> fe_values(fe, quadrature_formula,
                update_values | update_gradients |
                update_quadrature_points | update_JxW_values);

    const unsigned int   nd = fe.dofs_per_cell;
    const unsigned int   nq = quadrature_formula.size();

    // cell matrices
    FullMatrix<double>   Acell(nd,nd);
    FullMatrix<double>   Mcell(nd,nd);
    Vector<double>       Bcell(nd);
    std::vector<types::global_dof_index> local_dof_indices(nd);

    // texture and its gradient in the cell
    std::vector<double> texture_cell(nq);
    std::vector<Tensor<1,2> > tgrad_cell(nq);

    typename DoFHandler<dim>::active_cell_iterator cell;
    for (cell=dofs.begin_active(); cell!=dofs.end(); ++cell) {
      Acell = 0;
      Bcell = 0;
      Mcell = 0;

      fe_values.reinit(cell);
      fe_values.get_function_values(texture, texture_cell);
      fe_values.get_function_gradients(texture, tgrad_cell);

      for (unsigned int q=0; q<nq; ++q){
        for (unsigned int i=0; i<nd; ++i){
          for (unsigned int j=0; j<nd; ++j){
            Acell(i,j) +=(fe_values.shape_grad(i,q) *
                          fe_values.shape_grad(j,q) *
                          fe_values.JxW(q));
            if (!wave_nopot){
              Acell(i,j) -=(fe_values.shape_value(i,q) *
                            fe_values.shape_value(j,q) *
                            pow(sin(texture_cell[q]),2)*
                            fe_values.JxW(q));

              Acell(i,j) -=(fe_values.shape_value(i,q) *
                            fe_values.shape_value(j,q) *
                            tgrad_cell[q].norm_square() *
                            fe_values.JxW(q));
            }

            Mcell(i,j) +=(fe_values.shape_value(i,q) *
                          fe_values.shape_value(j,q) *
                          fe_values.JxW(q));

          }
          Bcell(i) += (fe_values.shape_value(i,q) *
                       0.1 * fe_values.JxW(q));
        }
      }
      // Finally, transfer the contributions from Acell and
      // Bcell into the global objects.
      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(
         Acell, Bcell, local_dof_indices, A, B);
      constraints.distribute_local_to_global(
         Mcell, local_dof_indices, M);
    }
  }


  // solve the system
  {
    en=-1;
    for (unsigned int i = 0; i<wave.size(); i++) wave[i]=1;

    SolverControl solver_control(1000, 1e-6);
    GrowingVectorMemory<> mem;
    EigenInverseM<>::AdditionalData data;
    data.relaxation = 0.1;
    data.use_residual = false;
    EigenInverseM<> solver(solver_control,mem,data);
    solver.solve(en, A, M, wave);
    std::cout << "En: " << en << "\n";

    constraints.distribute(wave);
  }

}

/*************************************************************************/
/*************************************************************************/

double
calc(double Lx, double Ly, double R){
  try {

    double grid_acc = 1e-3;
//    double text_acc = 1e-2;

    deallog.depth_console(0);
    PolarSolver ps(Lx, Ly, R);

    ps.make_initial_grid();
    ps.save_grid("grid1.eps");
    ps.do_text_calc();

    // loop for solving the texture equation and grid refinement
    double err=1;
    while (err>grid_acc){
      err = ps.refine_grid(0.3, 0.03);
      ps.do_text_calc();
    }
    std::cout << "grid accuracy after refinment: " << err << "\n";
    ps.save_grid("grid2.eps");

    // without grid refining
    for (int i=0; i<3; i++){
      ps.do_text_calc(false);
      //std::cout << ps.check_text() << "\n";
    }
    ps.save_data("text1.eps", ps.texture);

    ps.do_wave_calc();
    ps.save_data("wave1.eps", ps.wave);
    return ps.en;
  }
  catch(std::exception &exc) {
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl;
    return 0;
  }
  catch(...){
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl;
    return 0;
  }
}

int main(){
//  static const double DD[] = {0.20, 0.24, 0.28, 0.32, 0.36, 0.40, 0.44, 0.48, 0.52, 0.56, 0.60, 0.70, 0.80, 1.00,
//                              1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0, 8.00, 16.00, 20.00};

  static const double DD[] = {3.00};

  std::vector<double> D(DD, DD + sizeof(DD)/sizeof(DD[0]));
  std::vector<double> E;
  std::vector<double>::iterator i;
  for (i=D.begin(); i!=D.end(); i++){
    double R = (*i)/2.0;
    double Lx=4;
    double Ly=4;
    E.push_back(calc(Lx, Ly, R));
  }

  for (unsigned int i=0; i<D.size(); i++){
    printf("%5.2f %6.4f\n", D[i], -E[i]);
  }
  return 0;
}
