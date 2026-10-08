#include "container.hpp"
#include "cute/container/tuple.hpp"
#include <cute/tensor.hpp>

#include <cstdint>
#include <iostream>


void test_tuple() {
  using namespace cute;
  
  auto t = make_tuple(C<1>(), C<4>(), C<144>(), 5);

  // with zero-size optimization, t has size 4, the size of int.
  // The first 3 values has zero size, the last value has a dynamic 
  // size of 4. 
  std::cout<<sizeof(t)<<" expecting 4\n";

  // the cute tuple can do double direction mapping betewen
  // the tuple value type and the value type index.
  auto index = get<2>(t);
  std::cout<<index<<" expecting _144\n";

  // For reverse finding, it creates constant bool list by comparing 
  // the provided type with each tuple value type. 
  auto val_index = find<C<144>>(t);
  std::cout<<val_index<<" _2 \n ";

  auto dyn_val_index = find<int>(t);
  std::cout<<dyn_val_index<<" _3 \n ";
  
  // cute has a is_tuple trait type overload?  
  std::cout<<is_tuple<tuple<C<1>, C<2>> >::value << " expecting 1\n";
  std::cout<<is_tuple<C<1>>::value << " expecting 0\n";

  // now let's see the tuple_cat, which basically reconstruct 
  // a new tuple by constructing tuple element index with integer_sequence
  // then selecting each tuple element with get, and then reconstruct the 
  // resulting tuple with make_tuple.
  // let's test this api
  auto t1 = tuple<C<1>, C<4>, C<144>, int>{C<1>(), C<4>(), C<144>(), 5};
  auto t2 = make_tuple(C<2>{}, C<100>{});

  auto t3 = tuple_cat(t1, t2);
  std::cout<<t3<<"\n";
}