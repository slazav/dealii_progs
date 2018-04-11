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
#include "data_out_sym.h"
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
  enum BCType {
    HQV_PAIR_ZBC,
    HQV_PAIR_NBC,
    SQV_PAIR_ZBC,
    SQV_PAIR_NBC,
    SQV_CHAIN_ZBC,
    SQV_CHAIN_NBC
  } bctype;

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
  void save_data(const char* fname, const Vector<double> & data, int var=0);

  // Do the texture calculations.
  // repeat=true shows that texture didn't change after the last calculation
  void do_text_calc(bool repeat=false);

  // Do the wave calculation.
  void do_wave_calc();


  // calculate texture accuracy by integrating (nabla^2 a - sin(2a)/2)^2
  double check_text();

  // Calculate wave integrals.
  double calc_amp();

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

    bool wave_nopot;  // test mode -- no potential for the wave
};

PolarSolver::PolarSolver(
     const double Lx_, const double Ly_, const double R_):
     dofs(triang), fe(2), Lx(Lx_), Ly(Ly_), R(R_) {
  bctype=SQV_PAIR_NBC;
  wave_nopot = false;
}

PolarSolver::~PolarSolver() {
  dofs.clear();
  triang.clear();
}

/*************************************************************************/
// reset boundary IDs:
// 1: x=0 -- symmetry plane,         a'=0,  w'=0
// 2: y=0, x<R   -- between vortices a=pi/2 w'=0
// 3: y=0, x>=R  -- outside vortices a=0    w=0
// 4: x=Lx edge                      a=0 w=0 or a'=0 w'=0
// 0: y=Ly edge                      a=0 w=0 or a'=0 w'=0
void
PolarSolver::setup_boundary_ids(){
  typename Triangulation<dim>::cell_iterator
    cell = triang.begin(),
    endc = triang.end();
  for (; cell!=endc; ++cell){
    for (unsigned int i=0; i<GeometryInfo<dim>::faces_per_cell; i++){
      if (cell->face(i)->boundary_id() ==
          numbers::internal_face_boundary_id) continue;
      double xc = cell->face(i)->center()(0),
             yc = cell->face(i)->center()(1);
      if (R<Lx){ // finite soliton
        if (std::fabs(xc) < 1e-12)
          cell->face(i)->set_boundary_id(1);
        else if (std::fabs(xc) < R &&
                 std::fabs(yc) < 1e-12)
          cell->face(i)->set_boundary_id(2);
        else if (std::fabs(xc) >= R &&
                 std::fabs(yc) < 1e-12)
          cell->face(i)->set_boundary_id(3);
        else if (std::fabs(xc - Lx) < 1e-12)
          cell->face(i)->set_boundary_id(4);
        else
          cell->face(i)->set_boundary_id(0);
      }
      else { // infinite soliton
        if (std::fabs(xc) < 1e-12 ||
            std::fabs(xc - Lx) < 1e-12)
          cell->face(i)->set_boundary_id(1);
        else if (std::fabs(yc) < 1e-12)
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
PolarSolver::save_data(const char* fname, const Vector<double> & data, int var){
  DataOutBase::EpsFlags eps_flags;


  eps_flags.z_scaling = 2./data.linfty_norm();
//  eps_flags.draw_mesh = false;
  eps_flags.azimut_angle = 60;
  eps_flags.turn_angle   = 140;
  eps_flags.line_width   = 0.1;
  if (var>0) eps_flags.color_function = &sym_color_function;

  DataOutSym data_out;
  data_out.set_flags(eps_flags);

  data_out.attach_dof_handler(dofs);
  data_out.add_data_vector(data, "data");

  double xsh=2*M_PI;
  if (bctype==HQV_PAIR_ZBC || bctype==HQV_PAIR_NBC ||
      bctype==SQV_CHAIN_ZBC || bctype==SQV_CHAIN_NBC) xsh=M_PI;

  if (var==0) data_out.build_patches(0);
  else        data_out.build_patches(0, true, xsh);

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

    // for some reason order is important (should be same as in wave_calc?)
    switch (bctype){
      case HQV_PAIR_ZBC:
        VectorTools::interpolate_boundary_values(
            dofs, 0, ZeroFunction<dim>(), constraints);
        VectorTools::interpolate_boundary_values(
            dofs, 4, ZeroFunction<dim>(), constraints);
        VectorTools::interpolate_boundary_values(
            dofs, 3, ZeroFunction<dim>(), constraints);
        VectorTools::interpolate_boundary_values(
            dofs, 2, ConstantFunction<dim>(0.5*M_PI), constraints);
      break;
      case HQV_PAIR_NBC:
        VectorTools::interpolate_boundary_values(
            dofs, 3, ZeroFunction<dim>(), constraints);
        VectorTools::interpolate_boundary_values(
            dofs, 2, ConstantFunction<dim>(0.5*M_PI), constraints);
      break;
      case SQV_PAIR_ZBC:
        VectorTools::interpolate_boundary_values(
            dofs, 0, ZeroFunction<dim>(), constraints);
        VectorTools::interpolate_boundary_values(
            dofs, 4, ZeroFunction<dim>(), constraints);
        VectorTools::interpolate_boundary_values(
            dofs, 3, ZeroFunction<dim>(), constraints);
        VectorTools::interpolate_boundary_values(
            dofs, 2, ConstantFunction<dim>(M_PI), constraints);
      break;
      case SQV_PAIR_NBC:
        VectorTools::interpolate_boundary_values(
            dofs, 3, ZeroFunction<dim>(), constraints);
        VectorTools::interpolate_boundary_values(
            dofs, 2, ConstantFunction<dim>(M_PI), constraints);
      break;
      case SQV_CHAIN_ZBC:
        VectorTools::interpolate_boundary_values(
            dofs, 0, ZeroFunction<dim>(), constraints);
        VectorTools::interpolate_boundary_values(
            dofs, 3, ConstantFunction<dim>(-M_PI/2), constraints);
        VectorTools::interpolate_boundary_values(
            dofs, 2, ConstantFunction<dim>(M_PI/2), constraints);
      break;
      case SQV_CHAIN_NBC:
        VectorTools::interpolate_boundary_values(
            dofs, 0, ZeroFunction<dim>(), constraints);
        VectorTools::interpolate_boundary_values(
            dofs, 3, ConstantFunction<dim>(-M_PI/2), constraints);
        VectorTools::interpolate_boundary_values(
            dofs, 2, ConstantFunction<dim>(M_PI/2), constraints);
      break;
      default:
        std::cerr << "Unsupported vortex type\n";
        exit(1);
    }
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
  std::cerr << "  texture calculation: "
            << "  act.cells: " << std::setw(4) << triang.n_active_cells()
            << "  DOFs: " << std::setw(4) << dofs.n_dofs();


  // solve the system
  {
    SolverControl      solver_control(10000, 1e-12);
    SolverCG<>         solver(solver_control);

    PreconditionSSOR<> preconditioner;
    preconditioner.initialize(A, 1.2);
    solver.solve(A, texture, B, preconditioner);
    constraints.distribute(texture);
  }

  std::cerr << "  done" << std::endl;

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
// Calculate wave amplitude.
double
PolarSolver::calc_amp(){

  const QGauss<dim>  quadrature_formula(3);
  FEValues<dim> fe_values(fe, quadrature_formula,
              update_values | update_gradients | update_hessians |
              update_quadrature_points | update_JxW_values);

  const unsigned int   nq = quadrature_formula.size();

  // texture in the cell's quadrature points
  // same for laplacian
  std::vector<double> cell_text(nq), cell_wave(nq);

  double I0=0, I1=0, I2=0; // integrals
  typename DoFHandler<dim>::active_cell_iterator cell;
  for (cell= dofs.begin_active(); cell!=dofs.end(); ++cell) {
    fe_values.reinit(cell);
    fe_values.get_function_values(texture, cell_text);
    fe_values.get_function_values(wave, cell_wave);
    for (unsigned int q=0; q<nq; ++q){
      double wr = 2*cell_wave[q] * cos(-cell_text[q])
                + 2*cell_wave[q] * cos(-M_PI+cell_text[q]);
      double wi = 2*cell_wave[q] * sin(-cell_text[q])
                + 2*cell_wave[q] * sin(-M_PI+cell_text[q]);

      I0 += 4*cell_wave[q]*cell_wave[q]*fe_values.JxW(q);
      I1 += wr*fe_values.JxW(q);
      I2 += wi*fe_values.JxW(q);
    }
    std::cerr << "\n";
  }
  return (I1*I1 + I2*I2)/I0/(2*std::min(R, Lx));
}


/*************************************************************************/
// do the wave calculations
void
PolarSolver::do_wave_calc(){

  // set wave constraints
  constraints.clear();
  DoFTools::make_hanging_node_constraints(dofs, constraints);

  // for some reason order is important (should be same as in text_calc?)
  switch (bctype){
    case HQV_PAIR_ZBC:
      VectorTools::interpolate_boundary_values(
          dofs, 0, ZeroFunction<dim>(), constraints);
      VectorTools::interpolate_boundary_values(
          dofs, 4, ZeroFunction<dim>(), constraints);
      VectorTools::interpolate_boundary_values(
          dofs, 3, ZeroFunction<dim>(), constraints);
    break;
    case HQV_PAIR_NBC:
      VectorTools::interpolate_boundary_values(
          dofs, 3, ZeroFunction<dim>(), constraints);
    break;
    case SQV_PAIR_ZBC:
      VectorTools::interpolate_boundary_values(
          dofs, 0, ZeroFunction<dim>(), constraints);
      VectorTools::interpolate_boundary_values(
          dofs, 4, ZeroFunction<dim>(), constraints);
      VectorTools::interpolate_boundary_values(
          dofs, 3, ZeroFunction<dim>(), constraints);
    break;
    case SQV_PAIR_NBC:
      VectorTools::interpolate_boundary_values(
          dofs, 3, ZeroFunction<dim>(), constraints);
    break;
    case SQV_CHAIN_ZBC:
    break;
    case SQV_CHAIN_NBC:
    break;
    default:
      std::cerr << "Unsupported vortex type\n";
      exit(1);
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

    SolverControl solver_control(10000, 1e-6);
    GrowingVectorMemory<> mem;
    EigenInverseM<>::AdditionalData data;
    data.relaxation = 0.1;
    data.use_residual = false;
    EigenInverseM<> solver(solver_control,mem,data);
    solver.solve(en, A, M, wave);
    std::cerr << "En: " << en << "\n";

    constraints.distribute(wave);
  }

}

/*************************************************************************/
/*************************************************************************/

double
calc(double Lx, double Ly, double R){
  try {

//    double grid_acc = 1e-2;
//    double text_acc = 1e-2;

    deallog.depth_console(0);
    PolarSolver ps(Lx, Ly, R);

    ps.make_initial_grid();
    ps.save_grid("grid1.eps");
    ps.do_text_calc();

    // loop for solving the texture equation and grid refinement
    double err=1;
//    while (err>grid_acc){
    for (int i=0;i<8; i++) {
      err = ps.refine_grid(0.3, 0.03);
      ps.do_text_calc();
    }
    std::cerr << "grid accuracy after refinment: " << err << "\n";
    ps.save_grid("grid2.eps");

    // without grid refining
    for (int i=0; i<3; i++){
      ps.do_text_calc(false);
      //std::cerrt << ps.check_text() << "\n";
    }
    ps.save_data("text1.eps", ps.texture, 1);

    ps.do_wave_calc();
    ps.save_data("wave1.eps", ps.wave, 0);

    double amp = ps.calc_amp();
    fprintf(stdout, "%5.2f %5.2f %5.2f  %6.4f %6.4f\n", 2*Lx, 2*Ly, 2*R, ps.en, amp);

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

//  static const double DD[] = {0.2, 0.24, 0.28, 0.32, 0.36, 0.4, 0.44, 0.48, 0.52, 0.56, 0.60, 0.70, 0.80, 1.00,
//                              1.2, 1.4, 1.6, 1.8, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 9, 10,
//                              12, 16, 20, 25};
//  static const double DD[] = {1.75, 2.25, 2.75, 3.25};

  static const double DD[] = {8};

  std::vector<double> D(DD, DD + sizeof(DD)/sizeof(DD[0]));
  std::vector<double> E;
  std::vector<double>::iterator i;
  for (i=D.begin(); i!=D.end(); i++){
    double R = (*i)/2.0;
    double Lx=2*R;
    double Ly=R;

//    double R =  (*i)/2.0;
//    double Lx = (*i)/2.0;
//    double Ly= (*i)/2.0;

    E.push_back(calc(Lx, Ly, R));
  }

  return 0;
}
