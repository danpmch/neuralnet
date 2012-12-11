#include "perceptron.h"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include "util.h"

using namespace std;

Perceptron::Perceptron( int num_inputs ) : Neuron( num_inputs, THRESHOLD )
{
  for( int i = 0; i < get_weights().size(); i++ )
    get_weights()[ i ] = 0.0;
}

void Perceptron::train( vector< example > &examples, double error_thresh )
{
  cout << "Error threshold: " << error_thresh << endl;

  /*
  double iteration_error = error_thresh + 10;
  while( abs( iteration_error ) > error_thresh )
  {
    iteration_error = 0.0;

    for( int e = 0; e < examples.size(); e++ )
    {
      example current = examples[ e ];
      double a = activation( current.inputs );
      cout << "Computed activation: " << a << endl;

      double error = current.outputs[ 0 ] - a;
      cout << "Computed error: " << error << endl;
      iteration_error += error;

      add( current.inputs, error );
      cout << "New weights: "; print_vec( get_weights() );
    }
    iteration_error /= examples.size();

    cout << "Iteration error this round: " << iteration_error << endl;
  }
  */

  srand( time( NULL ) );
  while( abs( error( examples ) ) > error_thresh )
  {
    int e = rand() % examples.size();
    cout << "Using example " << e << endl;
    example current = examples[ e ];
    double a = activation( current.inputs );
    add( current.inputs, current.outputs[ 0 ] - a );
  }

}

double Perceptron::error( vector< example > &examples )
{
  double total_error = 0.0;
  for( int e = 0; e < examples.size(); e++ )
  {
    example current = examples[ e ];
    double a = activation( current.inputs );
    total_error += current.outputs[ 0 ] - a;
  }

  cout << "Total error: " << total_error << endl;
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
