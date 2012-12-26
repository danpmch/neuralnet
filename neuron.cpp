#include "neuron.h"

#include <stdexcept>
#include <cstdlib>
#include <cmath>
#include <iostream>

const double E = 2.71828182846;

double rand_clamped()
{
  double r = rand();
  return r / RAND_MAX / 10.0;
}

Neuron::Neuron( int num_inputs, bool zero )
{
//  srand( time( NULL ) );
  for( int i = 0; i < num_inputs + 1; i++ )
  {
    double val = zero ? 0.0 : rand_clamped();
    this->push_back( val );
  }
}

double Neuron::accumulate_activation( vector< double > inputs )
{
  if( inputs.size() != this->size() - 1 ) throw invalid_argument( "Input length doesn't match weight vector length" );

  double active = 0;
  int i;
  for( i = 0; i < ( int ) inputs.size(); i++ )
  {
    active += (*this)[ i ] * inputs[ i ];
  }
  active += (*this)[ i ];

  return active;
}

void Neuron::train( vector< example > &examples, double error_thresh )
{
  cout << "Error threshold: " << error_thresh << endl;

  double iterations = 0.0;
  double total_error = error_thresh + 10.0;
  srand( time( NULL ) );
  while( iterations++, total_error > error_thresh && iterations < 1000000 )
  {
    int e = rand() % examples.size();
    example &current = examples[ e ];
    double a = this->activation( current.inputs );
    update( current.inputs, current.outputs[ 0 ], a, alpha_factor( iterations ) );

    total_error = error( examples );
    cout << "Total error: " << total_error << "   Example: " << e << " Desired: " << current.outputs[ 0 ] << " Actual: " << a << " Error: " << current.outputs[ 0 ] - a << "\r";
  }

  cout << endl;
}

// total error is always positive
double Neuron::error( vector< example > &examples )
{
  double total_error = 0.0;
  for( int e = 0; e < ( int ) examples.size(); e++ )
  {
    example &current = examples[ e ];
    double a = this->activation( current.inputs );
    total_error += abs( current.outputs[ 0 ] - a );
  }

  return total_error / examples.size();
}

void Neuron::update( vector< double > &inputs, double desired, double actual, double alpha )
{
  double scale = this->scale_factor( desired, actual, alpha );

  int i;
  for( i = 0; i < ( int ) inputs.size(); i++ )
  {
    (*this)[ i ] += scale * inputs[ i ];
  }

  (*this)[ i ] += scale;
}

void SigmoidNeuron::update( vector< double > &inputs, double delta, double alpha )
{
  double scale = alpha * delta;

  int i;
  for( i = 0; i < ( int ) inputs.size(); i++ )
  {
    (*this)[ i ] += scale * inputs[ i ];
  }

  (*this)[ i ] += scale;
}

double SigmoidNeuron::compute_delta( double output, vector< SigmoidNeuron > &old_layer, double weight_index, vector< double > &old_deltas )
{
  double new_delta = 0.0;
  for( unsigned long neuron_index = 0; neuron_index < old_layer.size(); neuron_index++ )
  {
    SigmoidNeuron &neuron = old_layer[ neuron_index ];
    double old_delta = old_deltas[ neuron_index ];
    double weight = neuron[ weight_index ];

    new_delta += weight * old_delta;
  }

  double doutput = output * ( 1.0 - output );
  new_delta *= doutput;

  return new_delta;
}

double ThresholdNeuron::scale_factor( double desired_output, double actual_output, double alpha )
{
  return alpha * ( desired_output - actual_output );
}

double SigmoidNeuron::scale_factor( double desired_output, double actual_output, double alpha )
{
  return alpha * ( desired_output - actual_output ) * actual_output * ( 1.0 - actual_output );
}

double SigmoidNeuron::sigmoid( double x )
{
  double e_pow = 1.0 + pow( E, -x );
  return 1.0 / e_pow;
}

double SigmoidNeuron::dsigmoid( double x )
{
  double sig_x = sigmoid( x );
  return sig_x * ( 1.0 - sig_x );
}

