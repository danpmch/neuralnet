#ifndef __PERCEPTRON__
#define __PERCEPTRON__

#include "neuron.h"

struct example
{
  vector< double > inputs;
  vector< double > outputs;
};

template < int TOTAL_INPUTS >
class Perceptron : public ThresholdNeuron< TOTAL_INPUTS >
{

  public:

    Perceptron( int num_inputs );
    Perceptron( vector< double > &w ) : ThresholdNeuron< TOTAL_INPUTS >( w ) {};

    void train( vector< example > &examples, double error_thresh = 0.0 );

    double error( vector< example > &examples );

  private:

    void add( vector< double > &inputs, double scale );

};

#include "perceptron.cpp"

#endif
