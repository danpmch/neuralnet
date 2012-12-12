#include <stdexcept>
#include <cstdlib>
#include <cmath>

const double E = 2.71828182846;

double rand_clamped()
{
  double r = rand();
  return r / RAND_MAX;
}

template < int TOTAL_INPUTS >
Neuron< TOTAL_INPUTS >::Neuron( int num_inputs )
{
  srand( time( NULL ) );
  for( int i = 0; i < num_inputs + 1; i++ )
  {
    weights[ i ] = rand_clamped();
  }
}

template < int TOTAL_INPUTS >
Neuron< TOTAL_INPUTS >::Neuron( vector< double > &w )
{
  for( int i = 0; i < w.size(); i++ )
  {
    weights[ i ] = w[ i ];
  }
}

template < int TOTAL_INPUTS >
double Neuron< TOTAL_INPUTS >::accumulate_activation( vector< double > inputs )
{
  if( inputs.size() != TOTAL_INPUTS ) throw new invalid_argument( "Input length doesn't match weight vector length" );

  double active = 0;
  int i;
  for( i = 0; i < inputs.size(); i++ )
  {
    active += weights[ i ] * inputs[ i ];
  }
  active += weights[ i ];

  return active;
}

template < int TOTAL_INPUTS >
double SigmoidNeuron< TOTAL_INPUTS >::sigmoid( double x )
{
  double e_pow = 1.0 + pow( E, -x );
  return 1.0 / e_pow;
}
