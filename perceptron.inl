#include "perceptron.h"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include "util.h"

using namespace std;

template < int TOTAL_INPUTS >
Perceptron< TOTAL_INPUTS >::Perceptron()
{
  for( int i = 0; i < TOTAL_INPUTS + 1; i++ )
    this->get_weights()[ i ] = 0.0;
}

template < int TOTAL_INPUTS >
void Perceptron< TOTAL_INPUTS >::train( vector< example > &examples, double error_thresh )
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
template < int TOTAL_INPUTS >
double Perceptron< TOTAL_INPUTS >::error( vector< example > &examples )
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

template < int TOTAL_INPUTS >
void Perceptron< TOTAL_INPUTS >::add( vector< double > &inputs, double desired, double actual, double alpha )
{
  double scale = this->scale_factor( desired, actual, alpha );

  int i;
  for( i = 0; i < ( int ) inputs.size(); i++ )
  {
    this->get_weights()[ i ] += scale * inputs[ i ];
  }

  this->get_weights()[ i ] += scale;
}

template < int TOTAL_INPUTS >
double ThresholdPerceptron< TOTAL_INPUTS >::scale_factor( double desired_output, double actual_output, double alpha )
{
  return alpha * ( desired_output - actual_output );
}

template < int TOTAL_INPUTS >
double SigmoidPerceptron< TOTAL_INPUTS >::scale_factor( double desired_output, double actual_output, double alpha )
{
  return alpha * ( desired_output - actual_output ) * actual_output * ( 1.0 - actual_output );
}
