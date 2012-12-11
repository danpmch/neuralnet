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
      e.inputs.push_back( table[ row * 3 + col ] );
    }
    e.outputs.push_back( table[ row * 3 + cols - 1 ] );

    examples.push_back( e );
  }
}

void print_examples( vector< example > &examples )
{
  printf( "Examples: %d\n", examples.size() );
  for( int i = 0; i < examples.size(); i++ )
  {
    cout << "Example " << i << endl;
    cout << "  Inputs: "; print_vec( examples[ i ].inputs );
    cout << "  Output: "; print_vec( examples[ i ].outputs );
  }
}

void print_results( vector< example > &examples, Perceptron &p )
{
  cout << "Results:\n";
  for( int i = 0; i < examples.size(); i++ )
  {
    cout << "Example " << i << endl;
    cout << "  Inputs: "; print_vec( examples[ i ].inputs );
    cout << "  Output: " << p.activation( examples[ i ].inputs ) << endl;
  }

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

  vector< example > examples_2input;
  load_examples( and_truth_table, 4, 3, examples_2input );
  print_examples( examples_2input );

  vector< double > correct_weights;
  correct_weights.push_back( 1.0 );
  correct_weights.push_back( 1.0 );
  correct_weights.push_back( 2.0 );
  Perceptron p_correct( correct_weights );

  print_results( examples_2input, p_correct );

  Perceptron p_and( 2 );
  cout << "Training perceptron. Hold onto your butts...\n";
  p_and.train( examples_2input );
  cout << "Final perceptron weights: "; print_vec( p_and.get_weights() );
  cout << endl;

  print_results( examples_2input, p_and );
  examples_2input.clear();

  cout << "Training perceptron for OR:\n";
  load_examples( or_truth_table, 4, 3, examples_2input );
  print_examples( examples_2input );

  Perceptron p_or( 2 );
  p_or.train( examples_2input );
  cout << "Final perceptron weights: "; print_vec( p_or.get_weights() );
  print_results( examples_2input, p_or);

  return 0;
}
