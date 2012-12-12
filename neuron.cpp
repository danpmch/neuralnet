#include "neuron.h"
#include <stdexcept>
#include <cstdlib>
#include <cmath>

const double E = 2.71828182846;

double rand_clamped()
{
  double r = rand();
  return r / RAND_MAX;
}

Neuron::Neuron( int num_inputs, Function func )
{
  srand( time( NULL ) );
  for( int i = 0; i < num_inputs + 1; i++ )
  {
    weights.push_back( rand_clamped() );
  }

  type = func;
}

Neuron::Neuron( vector< double > &w, Function func )
{
  for( vector<double>::iterator i = w.begin(); i != w.end(); i++ )
  {
    weights.push_back( *i );
  }

  type = func;
}

double Neuron::accumulate_activation( vector< double > inputs )
{
  if( inputs.size() != weights.size() - 1 ) throw new invalid_argument( "Input length doesn't match weight vector length" );

  double active = 0;
  int i;
  for( i = 0; i < inputs.size(); i++ )
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
