#ifndef __PERCEPTRON__
#define __PERCEPTRON__

#include "neuron.h"

struct example
{
  vector< double > inputs;
  vector< double > outputs;
};

class Perceptron : public ThresholdNeuron
{

  public:

    Perceptron( int num_inputs );
    Perceptron( vector< double > &w ) : ThresholdNeuron( w ) {};

    void train( vector< example > &examples, double error_thresh = 0.0 );

    double error( vector< example > &examples );

  private:

    void add( vector< double > &inputs, double scale );

};

#endif
