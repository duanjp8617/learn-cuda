#include "util.hpp"
#include <cute/tensor.hpp>

#include <type_traits>
#include <cstdint>
#include <iostream>


void test_type_traits() {
  // type_traits redefines some type traits from stl.
  // A notable one is tuple_size and tuple_element_t.
  auto t = cute::tuple{1,2,3,4};
  auto t1 = cute::make_tuple(1,2,3,4,5);

  // Here, we try to test the tuple_size and tuple_element overload
  // from the cute library
  auto v = cute::tuple_size<decltype(t)>::value;
  std::cout<<v<<" \n";
  // While the tuple_size is basically an overloaded version
  // of std::tuple_size, it is still compatible with the 
  // cute::tuple type definition.

  // Let's try the original std library definition.
  auto v1 = std::tuple_size<decltype(t)>::value;
  std::cout<<v1<<" \n";
  // Of course it works.

  // Finally, let's try the std tuple
  auto std_t = std::make_tuple(1,2,3,4,5,6);
  auto std_t_v = cute::tuple_size<decltype(std_t)>::value;
  std::cout<<std_t_v<<" \n";
  // They are compatible as well.

  // Ok, here is tuple element t, I think I can handle it
  long long x = 100;
  int y= 1;
  int z = 2;
  auto et = cute::tuple{y,x,z};
  
  // tuple_element only inspect the type, it is possible just doing
  // a type-based recursion to retrive the corresponding type.
  // But for the actual value corresponding to the selected index, it can
  // not be acquired because cute tuple may have a different memeory layout
  // and member names as std tuple
  std::cout<<sizeof(cute::tuple_element<1, decltype(et)>::type)<< "\n";
  std::cout<<sizeof(cute::tuple_element<0, decltype(et)>::type)<< "\n";

  auto std_et = std::tuple{y, x, z};

  std::cout<<sizeof(cute::tuple_element<1, decltype(std_et)>::type)<< "\n";
  std::cout<<sizeof(cute::tuple_element<0, decltype(std_et)>::type)<< "\n";

}