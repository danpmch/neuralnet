#include <cstdio>
#include "perceptron.h"
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
    cout << "  Inputs: "; print_vec( examples[ i ].inputs );
    cout << "  Output: "; print_vec( examples[ i ].outputs );
  }
}

template <int S>
void print_results( vector< example > &examples, Perceptron<S> &p )
{
  cout << "Results:\n";
  for( int i = 0; i < ( int ) examples.size(); i++ )
  {
    cout << "Example " << i << endl;
    cout << "  Inputs: "; print_vec( examples[ i ].inputs );
    cout << "  Output: " << p.activation( examples[ i ].inputs ) << endl;
  }

}

template < int S >
void test_function( Perceptron< S > &p, double *table, int rows, int cols, double err_thresh = 0.0 )
{
  vector< example > examples;
  load_examples( table, rows, cols, examples);
  print_examples( examples);

  p.train( examples, err_thresh );
  cout << "Final perceptron weights: "; print_arr( p.get_weights(), p.total_weights() );
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

  vector< double > correct_weights;
  correct_weights.push_back( 1.0 );
  correct_weights.push_back( 1.0 );
  correct_weights.push_back( 2.0 );
  ThresholdPerceptron<2> p_correct( correct_weights );
  cout << "Correct weights: "; print_arr( p_correct.get_weights(), p_correct.total_weights() );
  print_results( examples, p_correct );
  cout << endl;

  cout << "Training a ThresholdPerceptron for AND:\n";
  ThresholdPerceptron<2> p_and;
  cout << "Original weights: "; print_arr( p_and.get_weights(), 3 );
  test_function( p_and, and_truth_table, 4, 3 );

  cout << "Training a ThresholdPerceptron for OR:\n";
  ThresholdPerceptron< 2 > p_or;
  test_function( p_or, or_truth_table, 4, 3 );

  /*
  cout << "Training a SigmoidPerceptron for OR:\n";
  SigmoidPerceptron<2> sig_or;
  test_function( sig_or, or_truth_table, 4, 3, 0.003 );
  */

  cout << "Training ThresholdPerceptron for NOT:\n";
  ThresholdPerceptron< 1 > p_not;
  test_function( p_not, not_truth_table, 2, 2 );

  /*
  cout << "Training a SigmoidPerceptron for NOT:\n";
  SigmoidPerceptron<1> sig_or;
  test_function( sig_or, not_truth_table, 2, 2, 0.003 );
  */

  /*
  cout << "Training a ThresholdPerceptron for XOR:\n";
  ThresholdPerceptron< 2 > p_xor;
  test_function( p_xor, xor_truth_table, 4, 3, 0.2 );
  */

  /*
  cout << "Training a SigmoidPerceptron for XOR:\n";
  SigmoidPerceptron< 2 > s_xor;
  test_function( s_xor, xor_truth_table, 4, 3, 0.253 );
  */

  cout << "Training a ThresholdPerceptron for Majority function of 3 inputs:\n";
  ThresholdPerceptron< 3 > p_maj;
  test_function( p_maj, marjoity_table, 8, 4 );

  return 0;
}
