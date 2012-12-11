#ifndef __UTIL__
#define __UTIL__

#include <cstdlib>
#include <iostream>

using namespace std;

template <class T>
void print_vec( vector< T > &vec )
{
  cout << "< ";
  for( int i = 0; i < vec.size(); i++ )
  {
    cout << vec[ i ];
    if( i < vec.size() - 1 )
      cout << ", ";
  }
  cout << " >" << endl;
}

#endif
