// this is my modification of the EigenInverse solver from deal.ii/eigen.h
//- use a mass matrix
//- use SolverCG instead of SolverGMRES solver

#ifndef eigen_inverse_m_h
#define eigen_inverse_m_h

#include <deal.II/base/config.h>
#include <deal.II/lac/shifted_matrix.h>
#include <deal.II/lac/solver.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/vector_memory.h>
#include <deal.II/lac/precondition.h>

#include <cmath>

DEAL_II_NAMESPACE_OPEN


/**
 * Inverse iteration (Wieland) for eigenvalue computations.
 *
 * This class implements an adaptive version of the inverse iteration by
 * Wieland.
 *
 * There are two choices for the stopping criterion: by default, the norm of
 * the residual $A x - l x$ is computed. Since this might not converge to zero
 * for non-symmetric matrices with non-trivial Jordan blocks, it can be
 * replaced by checking the difference of successive eigenvalues. Use
 * AdditionalData::use_residual for switching this option.
 *
 * Usually, the initial guess entering this method is updated after each step,
 * replacing it with the new approximation of the eigenvalue. Using a
 * parameter AdditionalData::relaxation between 0 and 1, this update can be
 * damped. With relaxation parameter 0, no update is performed. This damping
 * allows for slower adaption of the shift value to make sure that the method
 * converges to the eigenvalue closest to the initial guess. This can be aided
 * by the parameter AdditionalData::start_adaption, which indicates the first
 * iteration step in which the shift value should be adapted.
 *
 * @author Guido Kanschat, 2000, 2003
 */
template <class VECTOR = Vector<double> >
class EigenInverseM : private Solver<VECTOR>
{
public:
  /**
   * Declare type of container size.
   */
  typedef types::global_dof_index size_type;

  /**
   * Standardized data struct to pipe additional data to the solver.
   */
  struct AdditionalData
  {
    /**
     * Damping of the updated shift value.
     */
    double relaxation;

    /**
     * Start step of adaptive shift parameter.
     */
    unsigned int start_adaption;
    /**
     * Flag for the stopping criterion.
     */
    bool use_residual;
    /**
     * Constructor.
     */
    AdditionalData (double relaxation = 1.,
                    unsigned int start_adaption = 6,
                    bool use_residual = true)
      :
      relaxation(relaxation),
      start_adaption(start_adaption),
      use_residual(use_residual)
    {}

  };

  /**
   * Constructor.
   */
  EigenInverseM (SolverControl &cn,
                VectorMemory<VECTOR> &mem,
                const AdditionalData &data=AdditionalData());


  /**
   * Virtual destructor.
   */
  virtual ~EigenInverseM ();

  /**
   * Inverse method. @p value is the start guess for the eigenvalue and @p x
   * is the (not necessarily normalized, but nonzero) start vector for the
   * power method. After the iteration, @p value is the approximated
   * eigenvalue and @p x is the corresponding eigenvector, normalized with
   * respect to the l2-norm.
   */
  template <class MATRIX>
  void
  solve (double       &value,
         const MATRIX &A,
         const MATRIX &M,
         VECTOR       &x);

protected:
  /**
   * Flags for execution.
   */
  AdditionalData additional_data;
};

/*@}*/
//---------------------------------------------------------------------------


template <class VECTOR>
EigenInverseM<VECTOR>::EigenInverseM (SolverControl &cn,
                                    VectorMemory<VECTOR> &mem,
                                    const AdditionalData &data)
  :
  Solver<VECTOR>(cn, mem),
  additional_data(data)
{}



template <class VECTOR>
EigenInverseM<VECTOR>::~EigenInverseM ()
{}



template <class VECTOR>
template <class MATRIX>
void
EigenInverseM<VECTOR>::solve (double       &value,
                             const MATRIX &A,
                             const MATRIX &M,
                             VECTOR       &x)
{
  deallog.push("Wielandt");

  SolverControl::State conv=SolverControl::iterate;

  // Prepare matrix for solver
  ShiftedMatrixGeneralized <MATRIX,MATRIX,VECTOR> A_s(A, M, -value);

  // Define solver
  ReductionControl inner_control (5000, 1.e-16, 1.e-5, false, false);
  PreconditionIdentity prec;
//  SolverGMRES<VECTOR>  solver(inner_control, this->memory);
  SolverCG<VECTOR>  solver(inner_control);

  // Next step for recomputing the shift
  unsigned int goal = additional_data.start_adaption;

  // Auxiliary vector
  VECTOR *Vy = this->memory.alloc ();
  VECTOR &y = *Vy;
  y.reinit (x);
  VECTOR *Vr = this->memory.alloc ();
  VECTOR &r = *Vr;
  r.reinit (x);
  VECTOR *Vm = this->memory.alloc ();
  VECTOR &m = *Vm;
  m.reinit (x);
  VECTOR *Vx0 = this->memory.alloc ();
  VECTOR &x0 = *Vx0;
  x0.reinit (x);

  x0.equ(1.0,x);
  double length = x*x0;
  double old_value = value;
  x *= 1./length;

  // Main loop
  double res = -std::numeric_limits<double>::max();
  size_type iter=0;
  for (; conv==SolverControl::iterate; iter++)
    {
      M.vmult(m,x);

      solver.solve (A_s, y, m, prec);

      // Compute absolute value of eigenvalue
      M.vmult(m,y);
      length = m*x0;
      //std::cout << ">>" << iter << " " << A_s.shift() << " " << value << " " << 1./length << "\n";

      // Compute unshifted eigenvalue
      value = - A_s.shift() + 1./length;

      if (iter==goal)
        {
          const double new_shift = - additional_data.relaxation * value
                                   + (1.-additional_data.relaxation) * A_s.shift();
          A_s.shift(new_shift);
          ++goal;
        }

      // Update normalized eigenvector
      x.equ (1./length, y);
      // Compute residual
      if (additional_data.use_residual)
        {
          y.equ (value, x);
          A.vmult(r,x);
          M.vmult(m,x);
          r.sadd(-1., value, m);
          res = r.l2_norm();
          // Check the residual
          conv = this->iteration_status (iter, res, x);
        }
      else
        {
          res = std::fabs(1./value-1./old_value);
          conv = this->iteration_status (iter, res, x);
        }
      old_value = value;
    }

  this->memory.free(Vx0);
  this->memory.free(Vy);
  this->memory.free(Vr);
  this->memory.free(Vm);

  deallog.pop();

  // in case of failure: throw
  // exception
  AssertThrow (conv == SolverControl::success,
               SolverControl::NoConvergence (iter, res));
  // otherwise exit as normal
}

DEAL_II_NAMESPACE_CLOSE

#endif
