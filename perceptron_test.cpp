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

  ThresholdPerceptron<2> p_and;
  cout << "Original weights: "; print_arr( p_and.get_weights(), 3 );
  cout << "Training perceptron. Hold onto your butts...\n";
  p_and.train( examples);
  cout << "Final perceptron weights: "; print_arr( p_and.get_weights(), 3 );
  cout << endl;

  print_results( examples, p_and );
  examples.clear();

  cout << "Training perceptron for OR:\n";
  ThresholdPerceptron< 2 > p_or;
  test_function( p_or, or_truth_table, 4, 3 );

  /*
  cout << "Training a SigmoidPerceptron for OR:\n";
  SigmoidPerceptron<2> sig_or;
  sig_or.train( examples, 0.003 );
  cout << "Final perceptron weights: "; print_arr( sig_or.get_weights(), sig_or.total_weights() );
  print_results( examples, sig_or );
  */

  cout << "Training ThresholdPerceptron for NOT:\n";
  examples.clear();
  load_examples( not_truth_table, 2, 2, examples );
  print_examples( examples );
  ThresholdPerceptron<1> p_not;
  p_not.train( examples );
  cout << "Final perceptron weights: "; print_arr( p_not.get_weights(), p_not.total_weights() );
  print_results( examples, p_not );

  /*
  cout << "Training a SigmoidPerceptron for NOT:\n";
  SigmoidPerceptron<1> sig_or;
  sig_or.train( examples, 0.003 );
  cout << "Final perceptron weights: "; print_arr( sig_or.get_weights(), sig_or.total_weights() );
  print_results( examples, sig_or );
  */


  return 0;
}
