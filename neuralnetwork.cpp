#include "neuralnetwork.h"
#include "util.h"

#include <cstdlib>
#include <iostream>
#include <cmath>

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

  double total_error = error_thresh + 10.0;
  while( total_error > error_thresh )
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

      backpropagate( inputs, error );
      cout << "Inputs: "; print_vec( e->inputs ); cout << " Desired: "; print_vec( e->outputs ); cout << " Actual: "; print_vec( inputs[ inputs.size() - 1 ] ); 
      cout << " Error vec: "; print_vec( error ); cout << endl;

      print_net();
      cout << endl;
    }

    total_error = error( examples );
    cout << " Total error: " << total_error << "\n";
  }
//    cout << " Total error: " << total_error << "\n";
}

void NeuralNetwork::backpropagate( vector< vector< double > > &inputs, vector< double > err )
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
      current_layer[ neuron ].update( layer_inputs, neuron_output, err[ neuron ] );
    }

    // skip backpropagation if at last layer
    if( layer == 0 ) continue;

    // backpropagate the delta values for the next layer
    vector< double > new_err;
    for( unsigned long next_layer_neuron_index = 0; next_layer_neuron_index < network[ layer - 1 ].size(); next_layer_neuron_index++ )
    {
      new_err.push_back( 0.0 );
      for( unsigned long current_layer_neuron_index = 0; current_layer_neuron_index < current_layer.size(); current_layer_neuron_index++ )
      {
        double a = current_layer[ current_layer_neuron_index ].accumulate_activation( layer_inputs );
        new_err[ next_layer_neuron_index ] += 
          current_layer[ current_layer_neuron_index ][ next_layer_neuron_index ] * err[ current_layer_neuron_index ] * SigmoidNeuron::dsigmoid( a );
      }
    }
    err = new_err;

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

  return total_error / examples.size();
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
