#include "polar_solver.h"

//// Inter-vortex distances. Array with a few values:
//  static const double DD[] = {0.2, 0.24, 0.28, 0.32, 0.36, 0.4, 0.44, 0.48, 0.52, 0.56, 0.60, 0.70, 0.80, 1.00,
//                              1.2, 1.4, 1.6, 1.8, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 9, 10,
//                              12, 16, 20, 25};
//  static const double DD[] = {1.75, 2.25, 2.75, 3.25};
static const double DD[] = {8};
/*************************************************************************/

double
calc(double Lx, double Ly, double R, PolarSolver::BCType bctype){
  try {

//    double grid_acc = 1e-2;
//    double text_acc = 1e-2;

    //deallog.depth_console(0);
    PolarSolver ps(Lx, Ly, R, bctype);

    std::cerr << "Start from a coars grid (grid1.eps)\n";
    ps.make_initial_grid();
    ps.save_grid("grid1.eps");

    std::cerr << "Do a few texture calculations with the grid refinment:\n";
    double err=1;
//    while (err>grid_acc){
    for (int i=0;i<8; i++) {
      ps.do_text_calc();
      err = ps.refine_grid(0.3, 0.03); // refine 30% worst cells; coarse 3% best cells.
    }
    std::cerr << "grid accuracy after refinment (grid2.eps): " << err << "\n";
    ps.save_grid("grid2.eps");

    std::cerr << "Do a few texture calculations with the fixed grid:\n";
    for (int i=0; i<3; i++){
      ps.do_text_calc(false);
//      std::cerr << "Texture quality:" << ps.check_text() << "\n";
    }
    std::cerr << "Save the calculated texture to text1.eps.\n";
    ps.save_texture("text1.eps", 1);

    std::cerr << "Do wave calculation and save result to wave1.eps\n";
    ps.do_wave_calc();
    ps.save_wave("wave1.eps", 0);

    double amp = ps.get_amp();
    double en  = ps.get_en();
    std::cerr << "En: " << en << " Amp: " << amp << "\n";
    fprintf(stdout, "%5.2f %5.2f %5.2f  %6.4f %6.4f\n", 2*Lx, 2*Ly, 2*R, en, amp);

    return en;
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
  PolarSolver::BCType bctype = PolarSolver::HQV_PAIR_ZBC;

  std::vector<double> D(DD, DD + sizeof(DD)/sizeof(DD[0]));
  std::vector<double> E; // result
  std::vector<double>::iterator i;
  for (i=D.begin(); i!=D.end(); i++){
    double R = (*i)/2.0; // half of inter-vortex distance
    double Lx=4*R;       // calculation area, x-size
    double Ly=2*R;         // calculation area, y-size
    E.push_back(calc(Lx, Ly, R, bctype));
  }

  return 0;
}
