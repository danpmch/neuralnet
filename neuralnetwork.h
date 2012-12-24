#ifndef __NEURALNETWORK__
#define __NEURALNETWORK__

#include "neuron.h"

class NeuralNetwork
{
  public:

    NeuralNetwork( int total_inputs, vector< int > &structure );

    vector< double > compute( vector< double > &inputs );
    void full_compute( vector< vector< double > > &inputs );

    void train( vector< example > &examples, double error_thresh = 0.0 );
    double error( vector< example > &examples );

    void print_net();

    void set_neuron( int layer, int neuron, vector< double > &w );

    vector< SigmoidNeuron > & operator[]( int layer_index ) { return network[ layer_index ]; }

  private:

    vector< double > in_j( vector< SigmoidNeuron > &layer, vector< double > &inputs );
    void update_neuron( SigmoidNeuron &n, vector< double > &inputs, double output, double delta, double alpha = 1.0 );
    void backpropagate( vector< vector< double > > &inputs, vector< double > delta_j );

  private:

    vector< vector< SigmoidNeuron > > network;

};

#endif
