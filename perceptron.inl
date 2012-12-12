#include "perceptron.h"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include "util.h"

using namespace std;

template < int TOTAL_INPUTS >
Perceptron< TOTAL_INPUTS >::Perceptron( int num_inputs ) : ThresholdNeuron< TOTAL_INPUTS >( num_inputs )
{
  for( int i = 0; i < TOTAL_INPUTS + 1; i++ )
    this->get_weights()[ i ] = 0.0;
}

template < int TOTAL_INPUTS >
void Perceptron< TOTAL_INPUTS >::train( vector< example > &examples, double error_thresh )
{
  cout << "Error threshold: " << error_thresh << endl;

  srand( time( NULL ) );
  while( abs( error( examples ) ) > error_thresh )
  {
    int e = rand() % examples.size();
    cout << "Using example " << e << endl;
    example &current = examples[ e ];
    double a = this->activation( current.inputs );
    cout << "Desired: " << current.outputs[ 0 ] << " Actual: " << a << " Error: " << current.outputs[ 0 ] - a << endl;
    add( current.inputs, current.outputs[ 0 ] - a );
  }

}

template < int TOTAL_INPUTS >
double Perceptron< TOTAL_INPUTS >::error( vector< example > &examples )
{
  double total_error = 0.0;
  for( int e = 0; e < examples.size(); e++ )
  {
    example &current = examples[ e ];
    double a = this->activation( current.inputs );
    total_error += current.outputs[ 0 ] - a;
  }

  cout << "Total error: " << total_error / examples.size() << endl;
  return total_error / examples.size();
}

template < int TOTAL_INPUTS >
void Perceptron< TOTAL_INPUTS >::add( vector< double > &inputs, double scale )
{
  int i;
  for( i = 0; i < inputs.size(); i++ )
  {
    this->get_weights()[ i ] += scale * inputs[ i ];
  }

  this->get_weights()[ i ] += scale;
}
