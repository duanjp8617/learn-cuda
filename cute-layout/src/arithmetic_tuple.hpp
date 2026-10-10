#pragma once

#include <cute/numeric/arithmetic_tuple.hpp>

void test_arithmetic_tuple() {
  using namespace cute;
  // an arithmetic tuple is just a tuple
  auto a1 = ArithmeticTuple(make_tuple(1,2,3));
  print(a1);
  print("\n");

  print(is_tuple<decltype(a1)>::value);
  print("\n");

  print(is_flat<decltype(a1)>::value);
  print("\n");

  auto a2 = make_arithmetic_tuple(Int<1>{},Int<2>{},Int<3>{});
  print(a2);
  print("\n");

  auto a3 = as_arithmetic_tuple(make_tuple(1,2,3));
  print(a3);
  print("\n");

  // I'll leave ArithTupleIter here and examine it later.
  
  print(E<>{});
  print("\n");

  print(E<1>{});
  print("\n");

  print(E<0>{});
  print("\n");

  print(E<0,1>{});
  print("\n");

  print(as_arithmetic_tuple(E<1>{}));
  print("\n");

  print(as_arithmetic_tuple(make_basis_like(make_tuple(1,2,make_tuple(1,2)))));
  print("\n");

  auto a4 = make_arithmetic_tuple(Int<1>{}, tuple<Int<1>, Int<2>>{}, Int<3>{});
  print(get<0>(a4));
  print("\n");
  print(get<1>(a4));
  print("\n");
  print(a4);
  print("\n");

}