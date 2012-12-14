#ifndef __NEURON__
#define __NEURON__

#include <iostream>
#include <vector>
using namespace std;

struct example
{
  vector< double > inputs;
  vector< double > outputs;
};

class Neuron
{
  public:

    Neuron( int num_inputs );
    Neuron( vector< double > &w );

    vector< double > & get_weights() { return weights; }
    void set_weights( vector< double > &w ) { weights = w; }
    double & operator[]( int i ) { return weights[ i ]; }

    virtual double activation( vector< double > &inputs ) = 0;

  protected:

    double accumulate_activation( vector< double > inputs );

  private:

    vector< double > weights;

};

class ThresholdNeuron : public Neuron
{
  public:

    ThresholdNeuron( int num_inputs ) : Neuron( num_inputs ) { cout << "Initializing a ThresholdNeuron\n"; }
    ThresholdNeuron( vector< double > &w ) : Neuron( w ) {}

    virtual double activation( vector< double > &inputs ) { return threshold( this->accumulate_activation( inputs ) ); }
    inline double threshold( double x ) { if( x >= 0 ) return 1.0; else return 0.0; };
};

class SigmoidNeuron : public Neuron
{
  public:

    SigmoidNeuron( int num_inputs ) : Neuron( num_inputs ) { cout << "Initializing a SigmoidNeuron\n"; }
    SigmoidNeuron( vector< double > &w ) : Neuron( w ) {}

    virtual double activation( vector< double > &inputs ) { return sigmoid( this->accumulate_activation( inputs ) ); }
    double sigmoid( double x );
};

#endif
