#ifndef __UTIL__
#define __UTIL__

#include <cstdlib>
#include <iostream>

using namespace std;

template <class T>
void print_vec( vector< T > &vec )
{
  cout << "< ";
  for( int i = 0; i < ( int ) vec.size(); i++ )
  {
    cout << vec[ i ];
    if( i < ( int ) vec.size() - 1 )
      cout << ", ";
  }
  cout << " >";
}

template <class T>
void print_arr( T * arr, int size )
{
  cout << "< ";
  for( int i = 0; i < size; i++ )
  {
    cout << arr[ i ];
    if( i < size - 1 )
      cout << ", ";
  }
  cout << " >";
}

template <class T>
void prod( vector< T > &v1, const vector< T > &v2 )
{
  unsigned long min = v1.size() < v2.size() ? v1.size() : v2.size();
  for( unsigned long i = 0; i < min; i++ )
  {
    v1[ i ] *= v2[ i ];
  }
}

template <class T>
void add( vector< T > &v1, const vector< T > &v2 )
{
  unsigned long min = v1.size() < v2.size() ? v1.size() : v2.size();
  for( unsigned long i = 0; i < min; i++ )
  {
    v1[ i ] += v2[ i ];
  }
}

template <class T>
void scale( vector< T > &v1, double s )
{
  for( unsigned long i = 0; i < v1.size(); i++ )
  {
    v1[ i ] *= s;
  }
}


#endif
