#ifndef __NEURON__
#define __NEURON__

#include <vector>
using namespace std;

template < int TOTAL_INPUTS >
class Neuron
{
  public:

    Neuron( int num_inputs );
    Neuron( vector< double > &w );

    double * get_weights() { return weights; }

    virtual double activation( vector< double > &inputs ) = 0;

  protected:

    double accumulate_activation( vector< double > inputs );

  private:

    double weights[ TOTAL_INPUTS ];

};

template < int TOTAL_INPUTS >
class ThresholdNeuron : public Neuron< TOTAL_INPUTS >
{
  public:
    ThresholdNeuron( int num_inputs ) : Neuron< TOTAL_INPUTS >( num_inputs ) {}
    ThresholdNeuron( vector< double > &w ) : Neuron< TOTAL_INPUTS >( w ) {}

    virtual double activation( vector< double > &inputs ) { return threshold( this->accumulate_activation( inputs ) ); }
    inline double threshold( double x ) { if( x >= 0 ) return 1.0; else return 0.0; };
};

template < int TOTAL_INPUTS >
class SigmoidNeuron : public Neuron< TOTAL_INPUTS >
{
  public:
    SigmoidNeuron( int num_inputs ) : Neuron< TOTAL_INPUTS >( num_inputs ) {}
    SigmoidNeuron( vector< double > &w ) : Neuron< TOTAL_INPUTS >( w ) {}

    virtual double activation( vector< double > &inputs ) { return sigmoid( this->accumulate_activation( inputs ) ); }
    double sigmoid( double x );
};

#include "neuron.inl"

#endif
