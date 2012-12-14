#ifndef __PERCEPTRON__
#define __PERCEPTRON__

#include "neuron.h"

class Perceptron
{
  public:

    virtual void train( vector< example > &examples, double error_thresh = 0.0 ) = 0;
    virtual double error( vector< example > &examples ) = 0;

  private:

    virtual void add( vector< double > &inputs, double desired, double actual, double alpha = 1.0 ) = 0;

    virtual double alpha_factor( double iterations ) = 0;
    virtual double scale_factor( double desired_output, double actual_output, double alpha = 1.0 ) = 0;

};

class ThresholdPerceptron : public ThresholdNeuron, public Perceptron
{

  public:

    ThresholdPerceptron( int num_inputs );
    ThresholdPerceptron( vector< double > &w ) : ThresholdNeuron( w ) {};

    virtual void train( vector< example > &examples, double error_thresh = 0.0 );
    virtual double error( vector< example > &examples );

  private:

    virtual void add( vector< double > &inputs, double desired, double actual, double alpha = 1.0 );

    virtual double alpha_factor( double iterations ) { return 1000 / ( 1000 + iterations ); }
    virtual double scale_factor( double desired_output, double actual_output, double alpha = 1.0 );

};

class SigmoidPerceptron : public SigmoidNeuron, public Perceptron
{
  public:

    SigmoidPerceptron( int num_inputs );
    SigmoidPerceptron( vector< double > &w ) : SigmoidNeuron( w ) {};

    virtual void train( vector< example > &examples, double error_thresh = 0.0 );
    virtual double error( vector< example > &examples );

  private:

    virtual void add( vector< double > &inputs, double desired, double actual, double alpha = 1.0 );

    virtual double alpha_factor( double iterations ) { return 1.0; }
    virtual double scale_factor( double desired_output, double actual_output, double alpha = 1.0 );
};

#endif
