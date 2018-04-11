// output of symmentic data

#ifndef dealii__data_out_sym_h
#define dealii__data_out_sym_h


#include <deal.II/numerics/data_out.h>

DEAL_II_NAMESPACE_OPEN

class DataOutSym : public DataOut<2>
{
  public:

  virtual void build_patches (
        const unsigned int n_subdivisions = 0,
        const bool xinv=false,  // invert x<0 area
        const double xshift=0.0 // shift  x<0 area
       ){
    DataOut<2>::build_patches(n_subdivisions);

    unsigned int N=patches.size();
    patches.resize(4*N);

    for (unsigned int i=0; i<N;i++){

      // 4 patches from one
      patches[1*N+i] = patches[i];
      patches[2*N+i] = patches[i];
      patches[3*N+i] = patches[i];

      // swap path coordinates
      for (unsigned int j=0; j< GeometryInfo<2>::vertices_per_cell; j++){
        patches[1*N+i].vertices[j][0] = -patches[i].vertices[j][0];
        patches[2*N+i].vertices[j][1] = -patches[i].vertices[j][1];
        patches[3*N+i].vertices[j][0] = -patches[i].vertices[j][0];
        patches[3*N+i].vertices[j][1] = -patches[i].vertices[j][1];
      }

      if (xinv){
        for (unsigned int j=0; j < patches[i].data.size()[1]; j++){
          patches[2*N+i].data(0,j) = - patches[2*N+i].data(0,j);
          patches[3*N+i].data(0,j) = - patches[3*N+i].data(0,j);
        }
      }

      if (xshift!=0.0){
        for (unsigned int j=0; j < patches[i].data.size()[1]; j++){
          patches[2*N+i].data(0,j) = xshift + patches[2*N+i].data(0,j);
          patches[3*N+i].data(0,j) = xshift + patches[3*N+i].data(0,j);
        }
      }

    }
  }

//  virtual void build_patches (const Mapping<DH::dimension,DH::space_dimension> &mapping,
//                              const unsigned int n_subdivisions = 0,
//                              const CurvedCellRegion curved_region = curved_boundary);
};

DataOutBase::EpsFlags::RgbValues
sym_color_function(
    const double val, const double min, const double max){

  const double z = (max+min)/2;
  return val<z ?
    DataOutBase::EpsFlags::default_color_function(val, min, z):
    DataOutBase::EpsFlags::default_color_function(-val, -max, -z);
}

DEAL_II_NAMESPACE_CLOSE

#endif
