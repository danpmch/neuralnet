#include "perceptron.h"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include "util.h"

using namespace std;

Perceptron::Perceptron( int num_inputs ) : ThresholdNeuron( num_inputs )
{
  for( int i = 0; i < get_weights().size(); i++ )
    get_weights()[ i ] = 0.0;
}

void Perceptron::train( vector< example > &examples, double error_thresh )
{
  cout << "Error threshold: " << error_thresh << endl;

  srand( time( NULL ) );
  while( abs( error( examples ) ) > error_thresh )
  {
    int e = rand() % examples.size();
    cout << "Using example " << e << endl;
    example &current = examples[ e ];
    double a = activation( current.inputs );
    cout << "Desired: " << current.outputs[ 0 ] << " Actual: " << a << " Error: " << current.outputs[ 0 ] - a << endl;
    add( current.inputs, current.outputs[ 0 ] - a );
  }

}

double Perceptron::error( vector< example > &examples )
{
  double total_error = 0.0;
  for( int e = 0; e < examples.size(); e++ )
  {
    example &current = examples[ e ];
    double a = activation( current.inputs );
    total_error += current.outputs[ 0 ] - a;
  }

  cout << "Total error: " << total_error / examples.size() << endl;
  return total_error / examples.size();
}

void Perceptron::add( vector< double > &inputs, double scale )
{
  int i;
  for( i = 0; i < inputs.size(); i++ )
  {
    get_weights()[ i ] += scale * inputs[ i ];
  }

  get_weights()[ i ] += scale;
}
