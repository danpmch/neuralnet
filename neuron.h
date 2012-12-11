#ifndef __NEURON__
#define __NEURON__

#include <vector>
using namespace std;

class Neuron
{

  public:

    enum Function { SIGMOID, THRESHOLD };

  public:

    Neuron( int num_inputs, Function func = SIGMOID );
    Neuron( vector< double > &w, Function func = SIGMOID );

    vector< double > & get_weights() { return weights; }

    double activation( vector< double > &inputs );
    inline double sigmoid( double x );
    inline double threshold( double x ) { if( x > 0 ) return 1.0; else return 0.0; };

  private:

    Function type;
    vector< double > weights;

};

#endif
