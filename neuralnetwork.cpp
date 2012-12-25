#include "neuralnetwork.h"
#include "util.h"

#include <cstdlib>
#include <iostream>
#include <cmath>

vector< double > dsigmoid( vector< double > &x )
{
  vector< double > deriv;
  for( unsigned long i = 0; i < x.size(); i++ )
  {
    deriv.push_back( x[ i ] * ( 1.0 - x[ i ] ) );
  }

  return deriv;
}

NeuralNetwork::NeuralNetwork( int total_inputs, vector< int > &structure )
{
  for( int layer = 0; layer < ( int ) structure.size(); layer++ )
  {
    network.push_back( vector< SigmoidNeuron >() );

    vector< SigmoidNeuron > &current = network[ layer ];
    for( int neuron = 0; neuron < structure[ layer ]; neuron++ )
    {
      int inputs = layer != 0 ? structure[ layer - 1 ] : total_inputs;
      current.push_back( SigmoidNeuron( inputs ) );
    }
  }
}

vector< double > NeuralNetwork::compute( vector< double > &inputs )
{
  vector< double > old_output = inputs;
  vector< double > new_output;

  for( int layer = 0; layer < ( int ) network.size(); layer++ )
  {
    new_output.clear();

    for( int neuron = 0; neuron < ( int ) network[ layer ].size(); neuron++ )
    {
      new_output.push_back( network[ layer ][ neuron ].activation( old_output ) );
    }

    old_output = new_output;
  }

  return old_output;
}

// fills the inputs vector with the results of every computation in the network.
// note that the output of each layer is the input to the next, except for the last layer
void NeuralNetwork::full_compute( vector< vector< double > > &inputs )
{
  for( unsigned long layer = 0; layer < network.size(); layer++ )
  {
    vector< double > new_inputs;
    for( unsigned long neuron = 0; neuron < network[ layer ].size(); neuron++ )
    {
      new_inputs.push_back( network[ layer ][ neuron ].activation( inputs[ layer ] ) );
    }
    inputs.push_back( new_inputs );
  }
}

void NeuralNetwork::train( vector< example > &examples, double error_thresh )
{
  srand( time( NULL ) );
  while( error( examples ) > error_thresh )
  {
//    example &e = examples[ rand() % examples.size() ];

    for( vector< example >::iterator e = examples.begin(); e != examples.end(); e++ )
    {

      // compute error vector correct_output - actual_output
      vector< vector< double > > inputs;
      inputs.push_back( e->inputs );
      full_compute( inputs );

      vector< double > error = inputs[ inputs.size() - 1 ];
      scale( error, -1 );
      add( error, e->outputs );

      vector< double > in_k = in_j( network[ network.size() - 1 ], inputs[ inputs.size() - 2 ] );
      vector< double > delta_k = dsigmoid( in_k );
      prod( delta_k, error );
      
      backpropagate( inputs, delta_k );
    }
  }
}

void NeuralNetwork::backpropagate( vector< vector< double > > &inputs, vector< double > delta_j )
{
  for( long layer = ( long ) network.size() - 1; layer >= 0; layer-- )
  {
    vector< SigmoidNeuron > &current_layer = network[ layer ];
    vector< double > &layer_inputs = inputs[ layer ];
    vector< double > &layer_outputs = inputs[ layer + 1 ];

    // update the neurons in the current layer based on delta values and inputs
    for( unsigned long neuron = 0; neuron < current_layer.size(); neuron++ )
    {
      double neuron_output = layer_outputs[ neuron ];
      update_neuron( current_layer[ neuron ], layer_inputs, neuron_output, delta_j[ neuron ] );
    }

    // backpropagate the delta values for the next layer
    vector< double > next_layer_in = in_j( network[ layer - 1 ], inputs[ layer - 1 ] );
    vector< double > next_layer_din = dsigmoid( next_layer_in );
    vector< double > new_deltas;
    for( unsigned long next_layer_neuron_index = 0; next_layer_neuron_index < network[ layer - 1 ].size(); next_layer_neuron_index++ )
    {
      new_deltas.push_back( 0.0 );
      for( unsigned long current_layer_neuron_index = 0; current_layer_neuron_index < current_layer.size(); current_layer_neuron_index++ )
      {
        new_deltas[ next_layer_neuron_index ] += current_layer[ current_layer_neuron_index ][ next_layer_neuron_index ] * delta_j[ current_layer_neuron_index ];
      }
      new_deltas[ next_layer_neuron_index ] *= next_layer_din[ next_layer_neuron_index ];
    }
    delta_j = new_deltas;

  }
}

vector< double > NeuralNetwork::in_j( vector< SigmoidNeuron > &layer, vector< double > &inputs )
{
  vector< double > in;
  for( unsigned long neuron = 0; neuron < layer.size(); neuron++ )
  {
    SigmoidNeuron &n = layer[ neuron ];
    in.push_back( n.accumulate_activation( inputs ) );
  }

  return in;
}

void NeuralNetwork::update_neuron( SigmoidNeuron &n, vector< double > &inputs, double output, double delta, double alpha )
{
  double scale = alpha * output * delta;
  for( unsigned long i = 0; i < n.size(); i++ )
  {
    n[ i ] += scale * inputs[ i ];
  }
}

// comptutes the total L1 error over all examples
double NeuralNetwork::error( vector< example > &examples )
{
  double total_error = 0.0;
  for( vector< example >::iterator e = examples.begin(); e != examples.end(); e++ )
  {
    vector< double > error_vec = compute( e->inputs );
    scale( error_vec, -1 );
    add( error_vec, e->outputs );

    for( unsigned long i = 0; i < error_vec.size(); i++ )
    {
      total_error += abs( error_vec[ i ] );
    }
  }

  return total_error;
}

void NeuralNetwork::set_neuron( int layer, int neuron, vector< double > &w )
{
  network[ layer ][ neuron ] = w;
}

void NeuralNetwork::print_net()
{
  for( int layer = 0; layer < ( int ) network.size(); layer++ )
  {
    cout << layer << ": [ ";
    for( int neuron = 0; neuron < ( int ) network[ layer ].size(); neuron++ )
    {
      print_vec( network[ layer ][ neuron ] );
    }
    cout << "]\n";
  }
}
