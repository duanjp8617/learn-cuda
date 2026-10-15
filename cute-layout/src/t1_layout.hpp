#pragma once

#include <cute/layout.hpp>


void test_layout() {
  using namespace cute;

  Layout a = Layout<Shape<_4, Shape<_3, _6>>>{};
  print(a);
  print("\n");

  // This is implemented as a get<Is...>, which
  // is overloaded by sending the index Is... to
  // both the shape and the stride. Therefore
  // We get a new layout containing only the 0th
  // part of the original shape and stride.
  print(layout<0>(a));
  print("\n");

  Layout a1 = Layout<_3, _1>{};
  Layout b1 = Layout<_4, _3>{};
  // make_shape(shape0, shape1), make_stride(stride0, stride2)
  Layout row = make_layout(a1, b1);
  print(row);
  print("\n");

  print(make_layout(a1));
  print("\n");

  // make_layout can make infinitte many layouts together
  print(make_layout(a1, b1, make_layout(a1)));
  print("\n");

  // layout can be appended, which has similar behavior as make_layout
  print(append(a1, b1));
  print("\n");
  print(append<5>(a1, b1));
  print("\n");

  print(replace<1>(a, b1));
  print("\n");

  // It can also be grouped
  {
    Layout a = Layout<Shape<_2, _3, _5, _7>>{};
    Layout b = group<0,2>(a);
    print(b);
    print("\n");
    print(flatten(group<0,3>(a)));
    print("\n");
  }

  {
    Layout a = Layout<Shape<_4, Shape<_3, Shape<_2, _6>, _6>>>{};    
    auto res = slice(make_tuple(_, make_tuple(_, make_tuple( _,_1{}), _)), a);
    print(res);
    print("\n");
  }

  
}