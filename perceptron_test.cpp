#include <cstdio>
#include "neuron.h"
#include "util.h"
#include <iostream>
using namespace std;

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
  printf( "Examples: %lu\n", examples.size() );
  for( int i = 0; i < ( int ) examples.size(); i++ )
  {
    cout << "Example " << i << endl;
    cout << "  Inputs: "; print_vec( examples[ i ].inputs ); cout << endl;
    cout << "  Output: "; print_vec( examples[ i ].outputs ); cout << endl;
  }
}

void print_results( vector< example > &examples, Neuron &p )
{
  cout << "Results:\n";
  for( int i = 0; i < ( int ) examples.size(); i++ )
  {
    cout << "Example " << i << endl;
    cout << "  Inputs: "; print_vec( examples[ i ].inputs ); cout << endl;
    cout << "  Output: " << p.activation( examples[ i ].inputs ) << endl;
  }

}

void test_function( Neuron &p, double *table, int rows, int cols, double err_thresh = 0.0 )
{
  vector< example > examples;
  load_examples( table, rows, cols, examples);
  print_examples( examples);

  p.train( examples, err_thresh );
  cout << "Final perceptron weights: "; print_vec( p ); cout << endl;
  print_results( examples, p );
  cout << endl;
}

int main()
{
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

  vector< example > examples;
  load_examples( and_truth_table, 4, 3, examples);
  print_examples( examples);

  /*
  vector< double > correct_weights;
  correct_weights.push_back( 1.0 );
  correct_weights.push_back( 1.0 );
  correct_weights.push_back( 2.0 );
  ThresholdPerceptron p_correct( correct_weights );
  cout << "Correct weights: "; print_vec( p_correct ); cout << endl;
  print_results( examples, p_correct );
  cout << endl;
  */

  /*
  cout << "Training a ThresholdPerceptron for AND:\n";
  ThresholdPerceptron p_and( 2 );
  cout << "Original weights: "; print_vec( p_and ); cout << endl;
  test_function( p_and, and_truth_table, 4, 3 );
  */

  cout << "Training a SigmoidPerceptron for AND:\n";
  SigmoidNeuron p_and( 2, true );
  cout << "Original weights: "; print_vec( p_and ); cout << endl;
  test_function( p_and, and_truth_table, 4, 3, 0.003 );

  /*
  cout << "Training a ThresholdPerceptron for OR:\n";
  ThresholdPerceptron p_or( 2 );
  test_function( p_or, or_truth_table, 4, 3 );
  */

  cout << "Training a SigmoidPerceptron for OR:\n";
  SigmoidNeuron s_or( 2, true );
  test_function( s_or, or_truth_table, 4, 3, 0.003 );

  cout << "Training a SigmoidPerceptron for NOR:\n";
  SigmoidNeuron s_nor( 2, true );
  test_function( s_nor, nor_truth_table, 4, 3, 0.003 );

  /*
  cout << "Training ThresholdPerceptron for NOT:\n";
  ThresholdPerceptron p_not( 1 );
  test_function( p_not, not_truth_table, 2, 2 );
  */

  /*
  cout << "Training a SigmoidPerceptron for NOT:\n";
  SigmoidPerceptron s_not( 1 );
  test_function( s_not, not_truth_table, 2, 2, 0.003 );
  */

  /*
  cout << "Training a ThresholdPerceptron for XOR:\n";
  ThresholdPerceptron p_xor( 2 );
  test_function( p_xor, xor_truth_table, 4, 3, 0.2 );
  */

  /*
  cout << "Training a SigmoidPerceptron for XOR:\n";
  SigmoidPerceptron s_xor( 2 );
  test_function( s_xor, xor_truth_table, 4, 3, 0.253 );
  */

  /*
  cout << "Training a ThresholdPerceptron for Majority function of 3 inputs:\n";
  ThresholdPerceptron p_maj( 3 );
  test_function( p_maj, marjoity_table, 8, 4 );
  */

  return 0;
}
