#ifndef __PERCEPTRON__
#define __PERCEPTRON__

#include "neuron.h"

struct example
{
  vector< double > inputs;
  vector< double > outputs;
};

template < int TOTAL_INPUTS >
class Perceptron : public virtual Neuron< TOTAL_INPUTS >
{

  public:

    Perceptron();
    Perceptron( vector< double > &w ) : Neuron< TOTAL_INPUTS >( w ) {};

    void train( vector< example > &examples, double error_thresh = 0.0 );

    double error( vector< example > &examples );

  private:

    void add( vector< double > &inputs, double desired, double actual, double alpha = 1.0 );

    virtual double alpha_factor( double iterations ) = 0;
    virtual double scale_factor( double desired_output, double actual_output, double alpha = 1.0 ) = 0;

};

template < int TOTAL_INPUTS >
class ThresholdPerceptron : public ThresholdNeuron< TOTAL_INPUTS >, public Perceptron< TOTAL_INPUTS >
{
  public:

    ThresholdPerceptron() : ThresholdNeuron< TOTAL_INPUTS >(), Perceptron< TOTAL_INPUTS >() {}
    ThresholdPerceptron( vector< double > &w ) : ThresholdNeuron< TOTAL_INPUTS >( w ), Perceptron< TOTAL_INPUTS >( w ) {};

  private:

    virtual double alpha_factor( double iterations ) { return 1000 / ( 1000 + iterations ); }
    virtual double scale_factor( double desired_output, double actual_output, double alpha = 1.0 );
};

template < int TOTAL_INPUTS >
class SigmoidPerceptron : public SigmoidNeuron< TOTAL_INPUTS >, public Perceptron< TOTAL_INPUTS >
{
  public:

    SigmoidPerceptron() : SigmoidNeuron< TOTAL_INPUTS >(), Perceptron< TOTAL_INPUTS >() {}
    SigmoidPerceptron( vector< double > &w ) : SigmoidNeuron< TOTAL_INPUTS >( w ), Perceptron< TOTAL_INPUTS >( w ) {};

  private:

    virtual double alpha_factor( double iterations ) { return 1.0; }
    virtual double scale_factor( double desired_output, double actual_output, double alpha = 1.0 );
};

#include "perceptron.inl"

#endif
