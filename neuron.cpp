#include "neuron.h"

#include <stdexcept>
#include <cstdlib>
#include <cmath>
#include <iostream>

const double E = 2.71828182846;

double rand_clamped()
{
  double r = rand();
  return r / RAND_MAX;
}

Neuron::Neuron( int num_inputs )
{
  srand( time( NULL ) );
  for( int i = 0; i < num_inputs + 1; i++ )
  {
    weights.push_back( rand_clamped() );
  }
}

Neuron::Neuron( vector< double > &w )
{
  set_weights( w );
}

double Neuron::accumulate_activation( vector< double > inputs )
{
  if( inputs.size() != weights.size() - 1 ) throw invalid_argument( "Input length doesn't match weight vector length" );

  double active = 0;
  int i;
  for( i = 0; i < ( int ) inputs.size(); i++ )
  {
    active += weights[ i ] * inputs[ i ];
  }
  active += weights[ i ];

  return active;
}

double SigmoidNeuron::sigmoid( double x )
{
  double e_pow = 1.0 + pow( E, -x );
  return 1.0 / e_pow;
}
