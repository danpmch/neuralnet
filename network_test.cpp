#include "neuralnetwork.h"
#include "util.h"
#include <iostream>
#include <vector>

using namespace std;

double and_truth_table[] = { 0, 0, 0,
                             0, 1, 0,
                             1, 0, 0,
                             1, 1, 1 };

double or_truth_table[] =  { 0, 0, 0,
                             0, 1, 1,
                             1, 0, 1,
                             1, 1, 1 };

double nor_truth_table[] =  { 0, 0, 1,
                              0, 1, 1,
                              1, 0, 1,
                              1, 1, 0 };

double not_truth_table[] = { 0, 1,
                             1, 0 };

double xor_truth_table[] = { 0, 0, 0,
                             0, 1, 1,
                             1, 0, 1,
                             1, 1, 0 };

double marjoity_table[] = { 0, 0, 0, 0,
                            0, 0, 1, 0,
                            0, 1, 0, 0,
                            0, 1, 1, 1,
                            1, 0, 0, 0,
                            1, 0, 1, 1,
                            1, 1, 0, 1,
                            1, 1, 1, 1 };


void load_examples( double *table, int rows, int cols, vector< example > &examples )
{
  for( int row = 0; row < rows; row++ )
  {
    example e;
    for( int col = 0; col < cols - 1; col++ )
    {
      e.inputs.push_back( table[ row * cols + col ] );
    }
    e.outputs.push_back( table[ row * cols + cols - 1 ] );

    examples.push_back( e );
  }
}

void print_examples( vector< example > &examples )
{
  cout << "Examples: " << examples.size() << endl;
  for( int i = 0; i < ( int ) examples.size(); i++ )
  {
    cout << "Example " << i << endl;
    cout << "  Inputs: "; print_vec( examples[ i ].inputs ); cout << endl;
    cout << "  Output: "; print_vec( examples[ i ].outputs ); cout << endl;
  }
}

void print_results( vector< example > &examples, NeuralNetwork &net )
{
  cout << "Network:\n";
  net.print_net();
  cout << endl;

  cout << "Results:\n";
  for( int i = 0; i < ( int ) examples.size(); i++ )
  {
    vector< double > outputs = net.compute( examples[ i ].inputs );

    cout << "Example " << i << endl;
    cout << "  Inputs: "; print_vec( examples[ i ].inputs ); cout << endl;
    cout << "  Output: "; print_vec( outputs ); cout << endl;
  }

}

void test_function( NeuralNetwork &net, double *table, int rows, int cols, double err_thresh = 0.0 )
{
  vector< example > examples;
  load_examples( table, rows, cols, examples);
  print_examples( examples);

  net.train( examples, err_thresh );
  print_results( examples, net );
  cout << endl;
}

void test_explicit_xor_network()
{
  // manually create a neural network for XOR
  vector< int > shape;
  shape.push_back( 2 );
  shape.push_back( 1 );
  NeuralNetwork net( 2, shape );

  vector< double > or_weights;
  or_weights.push_back( 10.9337 );
  or_weights.push_back( 10.9278 );
  or_weights.push_back( -5.2572 );

  vector< double > nor_weights;
  nor_weights.push_back( -10.9859 );
  nor_weights.push_back( -10.9861 );
  nor_weights.push_back( 16.5616 );

  vector< double > and_weights;
  and_weights.push_back( 10.9868 );
  and_weights.push_back( 10.984 );
  and_weights.push_back( -16.5618 );

  net[ 0 ][ 0 ] = or_weights;
  net[ 0 ][ 1 ] = nor_weights;
  net[ 1 ][ 0 ] = and_weights;

  net.print_net();

  for( int first = 0; first < 2; first++ )
  {
    for( int second = 0; second < 2; second++ )
    {
      vector< double > inputs;
      inputs.push_back( first );
      inputs.push_back( second );

      vector< vector< double > > compute_record;
      compute_record.push_back( inputs );
      net.full_compute( compute_record );

      vector< double > result = compute_record[ compute_record.size() - 1 ];
      print_vec( inputs ); cout << ": "; print_vec( result ); cout << endl;
    }
  }
}

int main()
{
  test_explicit_xor_network();

  vector< int > structure;
  structure.push_back( 1 );

  /*
  cout << "Training not: \n";
  NeuralNetwork not_net( 1, structure );
  test_function( not_net, not_truth_table, 2, 2, 0.003 );

  cout << "Training and: \n";
  NeuralNetwork and_net( 2, structure );
  test_function( and_net, and_truth_table, 4, 3, 0.003 );

  cout << "Training or: \n";
  NeuralNetwork or_net( 2, structure );
  test_function( or_net, or_truth_table, 4, 3, 0.003 );

  cout << "Training nor: \n";
  NeuralNetwork nor_net( 2, structure );
  test_function( nor_net, nor_truth_table, 4, 3, 0.003 );
  */

  cout << "Training xor: \n";

  structure.clear();
  structure.push_back( 2 );
  structure.push_back( 1 );

  NeuralNetwork xor_net( 2, structure );
  test_function( xor_net, xor_truth_table, 4, 3, 0.003 );

  return 0;
}

