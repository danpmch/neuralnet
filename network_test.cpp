#include <iostream>
#include "neuralnetwork.h"
#include <vector>
#include "util.h"

using namespace std;

int main()
{

  // create a neural network for XOR
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

  return 0;
}

