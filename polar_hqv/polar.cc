#include <getopt.h> // for getopt
#include <cstring>
#include "polar_solver.h"

// print help
void help(){
  std::cout <<
  "polar - calculate testure and spin waves in the polar phase of 3He.\n"
  "Usage: polar [options]\n"
  "Options:\n"
  " -h         write this help message and exit\n"
  " -R <value>  1/2 of the inter-vortex distance, in \\xi units (default 4)\n"
  " -X <value>  1/2 of the X-dimension\n"
  " -Y <value>  1/2 of the Y-dimension\n"
  " -x <value>  ratio X/R (default 2, used if -X is not set)\n"
  " -y <value>  ratio Y/R (default 1, used if -Y is not set)\n"

  "Calculation is done in X x Y area, then result is symmetrically\n"
  "extended to 2X x 2Y. Use R>X for infinite soliton.\n"
  "See bctype.png image. \n"
  " -e <value>  initial value for energy (default -1)\n"
  " -g <value>  grid accuracy (default 1e-3)\n"
  " -G <fname>  EPS file for grid image    (default grid.eps)\n"
  " -T <fname>  EPS file for texture image (default text.eps)\n"
  " -W <fname>  EPS file for wave image    (default wave.eps)\n"
  " -w (0|1)    draw mesh on the wave plot (default 0)\n"
  " -r (0|1)    rotated wave plot (default 0)\n"
  ;
}

int main(int argc, char *argv[]){
  try {
    /* default values */
    double R  = 4.0; // 1/2 of intervortex distance
    double Lx = 0.0; // 1/2 of the calculation area x size
    double Ly = 0.0; // 1/2 of the calculation area y size
    double kx = 2.0;
    double ky = 1.0;
    PolarSolver::BCType bctype = PolarSolver::HQV_PAIR_ZBC;
    double grid_acc = 1e-3; // grid accuracy
    const char *fname_w = "wave.eps";
    const char *fname_t = "text.eps";
    const char *fname_g = "grid.eps";
    double en0 = -1; // initial value for energy
    bool wave_mesh = false; // draw mesh on the wave plot 
    bool wave_rot  = false; // rotated wave plot

    /* parse  options */
    opterr=0;
    while(1){
      int c = getopt(argc, argv, "+hR:X:Y:x:y:e:g:G:T:W:w:r:");
      if (c==-1) break;
      switch (c){
        case '?': throw Err() << "Unknown option: -" << (char)optopt;
        case ':': throw Err() << "No argument: -" << (char)optopt;
        case 'h': help(); return 0;
        case 'R': R  = atof(optarg); break;
        case 'X': Lx = atof(optarg); break;
        case 'Y': Ly = atof(optarg); break;
        case 'x': kx = atof(optarg); break;
        case 'y': ky = atof(optarg); break;
        case 'e': en0 = atof(optarg); break;
        case 'g': grid_acc = atof(optarg); break;
        case 'G': fname_g = optarg; break;
        case 'T': fname_t = optarg; break;
        case 'W': fname_w = optarg; break;
        case 'w': wave_mesh = atoi(optarg); break;
        case 'r': wave_rot  = atoi(optarg); break;
      }
    }
    argc-=optind;
    argv+=optind;
    optind=1;

    if (R  <= 0.0) throw Err() << "R should be positive";
    if (Lx <= 0.0) Lx = kx*R;
    if (Ly <= 0.0) Ly = ky*R;

//    double text_acc = 1e-2;

    PolarSolver ps(Lx, Ly, R, bctype);


    std::cerr << "Do a few texture calculations with the grid refinment:\n";
    double err=1;
    ps.make_initial_grid();
    ps.do_text_calc();
    while (err>grid_acc){
      err = ps.refine_grid(0.3, 0.03); // refine 30% worst cells; coarse 3% best cells.
      ps.do_text_calc(false);
      //std::cerr << "Texture quality:" << ps.check_text() << "\n";
    }
    std::cerr << "Grid accuracy after refinment: " << err << "\n";
    if (fname_g != NULL && strlen(fname_g)>0){
      std::cerr << "Saving grid to " << fname_g << "\n";
      ps.save_grid(fname_g);
    }
    if (fname_t != NULL && strlen(fname_t)>0){
      std::cerr << "Saving texture to " << fname_t << "\n";
      ps.save_texture(fname_t);
    }

    std::cerr << "Do wave calculation\n";
    ps.do_wave_calc(en0);

    if (fname_w != NULL && strlen(fname_w)>0){
      std::cerr << "Saving wave to " << fname_w << "\n";
      ps.save_wave(fname_w, wave_mesh, wave_rot);
    }

    double amp = ps.get_amp();
    double en  = ps.get_en();
    std::cerr << "En: " << en << " Amp: " << amp << "\n";
    fprintf(stdout, "%5.2f %5.2f %5.2f  %e %e\n", 2*Lx, 2*Ly, 2*R, en, amp);

  }
  catch (Err E){
    std::cerr << "Error: " << E.str() << std::endl;
  }
  catch(std::exception &exc) {
    std::cerr << "Error: " << exc.what() << std::endl;
  }
  catch(...){
    std::cerr << "Unknown Error!" << std::endl;
  }
  return 0;
}
