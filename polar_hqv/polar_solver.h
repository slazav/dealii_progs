#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/constraint_matrix.h>

// 2-dimensional problem
#define DIM 2

// Error class
class Err {
  std::ostringstream s;
  public:
    Err(){}
    Err(const Err & o) { s << o.s.str(); }
    template <typename T>
    Err & operator<<(const T & o){ s << o; return *this; }
    std::string str()   const { return s.str(); }
};

/*************************************************************************/
class PolarSolver {
  public:


  // Possible types of boundary conditions (see bctype.png image).
  enum BCType {
    HQV_PAIR_ZBC,
    HQV_PAIR_NBC,
    SQV_PAIR_ZBC,
    SQV_PAIR_NBC,
    SQV_CHAIN_ZBC,
    SQV_CHAIN_NBC
  };

  // create a solver. Grid size is Lx x Ly, half of inter-vortex distance is R
  // (see bctype.png image).
  PolarSolver(const double Lx, const double Ly, const double R, const BCType bctype);
  ~PolarSolver();

  // Make the initial grid and initialize the texture to zero.
  void make_initial_grid();

  // Refine the grid according with the texture gradients
  // and redistribute the texture in the new grid.
  //  <ref> - amount of worse cells to be refined (0..1)
  //  <crs> - amount of best cells to be coarsed (0..1)
  double refine_grid(const double ref, const double crs);

  // save the grid to eps file
  void save_grid(const char* fname);

  // save the texture to eps file
  void save_texture(const char* fname, bool draw_mesh=true, bool draw_rot=true) {
    save_data(fname, texture, 1, draw_mesh, draw_rot);}

  // save the wave to eps file
  void save_wave(const char* fname, bool draw_mesh=true, bool draw_rot=true) {
    save_data(fname, wave, 0, draw_mesh, draw_rot);}

  // Do the texture calculations.
  // repeat=true shows that texture didn't change after the last calculation
  void do_text_calc(bool repeat=false);

  // Do the wave calculation. Parameter en is the initial value for energy.
  void do_wave_calc(double en=-1);

  // Calculate texture accuracy by integrating (nabla^2 a - sin(2a)/2)^2
  // (Does not work?)
  double check_text();

  // return integral IM of the calculated wave.
  double get_amp();

  // return eigenvalue of the calculated wave.
  double get_en() {return en;};

  private:

    // save a data vector to eps file
    void save_data(const char* fname, const dealii::Vector<double> & data, int var,
                   bool draw_mesh, bool draw_rot);

    // Setup boundary IDs. Used in make_initial_grid() and refine_grid().
    void setup_boundary_ids();

    dealii::Vector<double> texture;
    dealii::Vector<double> wave;
    double                 en;

    dealii::Triangulation<DIM>     triang;
    dealii::DoFHandler<DIM>        dofs;
    dealii::FE_Q<DIM>              fe;
    dealii::ConstraintMatrix     constraints;
    dealii::SparsityPattern      sparsity;
    double Lx, Ly, R;
    BCType bctype;

    dealii::SparseMatrix<double> A,M;
    dealii::Vector<double>       B;

    bool wave_nopot;  // test mode -- no potential for the wave
};
