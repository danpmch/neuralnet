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

class Neuron : public vector< double >
{
  public:

    Neuron( int num_inputs, bool zero = false );
    Neuron( vector< double > &w ) : vector< double >( w ) {}

    double accumulate_activation( vector< double > inputs );
    virtual double activation( vector< double > &inputs ) = 0;

    virtual void train( vector< example > &examples, double error_thresh = 0.0 );
    virtual double error( vector< example > &examples );

    virtual void update( vector< double > &inputs, double desired, double actual, double alpha = 1.0 );

  private:

    virtual double alpha_factor( double iterations ) = 0;
    virtual double scale_factor( double desired_output, double actual_output, double alpha = 1.0 ) = 0;

};

class ThresholdNeuron : public Neuron
{
  public:

    ThresholdNeuron( int num_inputs, bool zero = false ) : Neuron( num_inputs, zero ) {  }
    ThresholdNeuron( vector< double > &w ) : Neuron( w ) {}

    virtual double activation( vector< double > &inputs ) { return threshold( this->accumulate_activation( inputs ) ); }
    inline double threshold( double x ) { if( x >= 0 ) return 1.0; else return 0.0; };

  private:

    virtual double alpha_factor( double iterations ) { return 1000 / ( 1000 + iterations ); }
    virtual double scale_factor( double desired_output, double actual_output, double alpha = 1.0 );
};

class SigmoidNeuron : public Neuron
{
  public:

    SigmoidNeuron( int num_inputs, bool zero = false ) : Neuron( num_inputs, zero ) {  }
    SigmoidNeuron( vector< double > &w ) : Neuron( w ) {}

    virtual double activation( vector< double > &inputs ) { return sigmoid( this->accumulate_activation( inputs ) ); }

    virtual void update( vector< double > &inputs, double output, double err, double alpha = 1.0 );

    static double sigmoid( double x );
    static double dsigmoid( double x );

  private:

    virtual double alpha_factor( double iterations ) { return 1.0; }
    virtual double scale_factor( double desired_output, double actual_output, double alpha = 1.0 );
};

#endif
