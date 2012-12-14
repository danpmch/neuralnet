#include "perceptron.h"

#include <iostream>
#include <cmath>
#include <cstdlib>
#include "util.h"

using namespace std;

/*************************   THRESHOLD  ***********************************************************************/

ThresholdPerceptron::ThresholdPerceptron( int num_inputs ) : ThresholdNeuron( num_inputs )
{
  cout << "Initializing a ThresholdPerceptron\n";
  for( int i = 0; i < ( int ) this->get_weights().size(); i++ )
    this->get_weights()[ i ] = 0.0;
}

void ThresholdPerceptron::train( vector< example > &examples, double error_thresh )
{
  cout << "Error threshold: " << error_thresh << endl;

  double iterations = 0.0;
  srand( time( NULL ) );
  while( iterations++, error( examples ) > error_thresh && iterations < 1000000 )
  {
    int e = rand() % examples.size();
    cout << "Using example " << e << endl;
    example &current = examples[ e ];
    double a = this->activation( current.inputs );
    cout << "Desired: " << current.outputs[ 0 ] << " Actual: " << a << " Error: " << current.outputs[ 0 ] - a << endl;
    add( current.inputs, current.outputs[ 0 ], a, alpha_factor( iterations ) );
  }

}

// total error is always positive
double ThresholdPerceptron::error( vector< example > &examples )
{
  double total_error = 0.0;
  for( int e = 0; e < ( int ) examples.size(); e++ )
  {
    example &current = examples[ e ];
    double a = this->activation( current.inputs );
    total_error += abs( current.outputs[ 0 ] - a );
  }

  cout << "Total error: " << total_error / examples.size() << endl;
  return total_error / examples.size();
}

void ThresholdPerceptron::add( vector< double > &inputs, double desired, double actual, double alpha )
{
  double scale = this->scale_factor( desired, actual, alpha );

  int i;
  for( i = 0; i < ( int ) inputs.size(); i++ )
  {
    this->get_weights()[ i ] += scale * inputs[ i ];
  }

  this->get_weights()[ i ] += scale;
}

double ThresholdPerceptron::scale_factor( double desired_output, double actual_output, double alpha )
{
  return alpha * ( desired_output - actual_output );
}

/*************************   SIGMOID  ***********************************************************************/

SigmoidPerceptron::SigmoidPerceptron( int num_inputs ) : SigmoidNeuron( num_inputs )
{
  for( int i = 0; i < ( int ) this->get_weights().size(); i++ )
    this->get_weights()[ i ] = 0.0;
}

void SigmoidPerceptron::train( vector< example > &examples, double error_thresh )
{
  cout << "Error threshold: " << error_thresh << endl;

  double iterations = 0.0;
  srand( time( NULL ) );
  while( iterations++, error( examples ) > error_thresh && iterations < 1000000 )
  {
    int e = rand() % examples.size();
    cout << "Using example " << e << endl;
    example &current = examples[ e ];
    double a = this->activation( current.inputs );
    cout << "Desired: " << current.outputs[ 0 ] << " Actual: " << a << " Error: " << current.outputs[ 0 ] - a << endl;
    add( current.inputs, current.outputs[ 0 ], a, alpha_factor( iterations ) );
  }

}

// total error is always positive
double SigmoidPerceptron::error( vector< example > &examples )
{
  double total_error = 0.0;
  for( int e = 0; e < ( int ) examples.size(); e++ )
  {
    example &current = examples[ e ];
    double a = this->activation( current.inputs );
    total_error += abs( current.outputs[ 0 ] - a );
  }

  cout << "Total error: " << total_error / examples.size() << endl;
  return total_error / examples.size();
}

void SigmoidPerceptron::add( vector< double > &inputs, double desired, double actual, double alpha )
{
  double scale = this->scale_factor( desired, actual, alpha );

  int i;
  for( i = 0; i < ( int ) inputs.size(); i++ )
  {
    this->get_weights()[ i ] += scale * inputs[ i ];
  }

  this->get_weights()[ i ] += scale;
}

double SigmoidPerceptron::scale_factor( double desired_output, double actual_output, double alpha )
{
  return alpha * ( desired_output - actual_output ) * actual_output * ( 1.0 - actual_output );
}
