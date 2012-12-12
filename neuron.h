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

    virtual double activation( vector< double > &inputs ) = 0;

  protected:

    double accumulate_activation( vector< double > inputs );

  private:

    Function type;
    vector< double > weights;

};

class ThresholdNeuron : public Neuron
{
  public:
    ThresholdNeuron( int num_inputs ) : Neuron( num_inputs ) {}
    ThresholdNeuron( vector< double > &w ) : Neuron( w ) {}

    virtual double activation( vector< double > &inputs ) { return threshold( accumulate_activation( inputs ) ); }
    inline double threshold( double x ) { if( x >= 0 ) return 1.0; else return 0.0; };
};

class SigmoidNeuron : public Neuron
{
  public:
    SigmoidNeuron( int num_inputs ) : Neuron( num_inputs ) {}
    SigmoidNeuron( vector< double > &w ) : Neuron( w ) {}

    virtual double activation( vector< double > &inputs ) { return sigmoid( accumulate_activation( inputs ) ); }
    double sigmoid( double x );
};

#endif
