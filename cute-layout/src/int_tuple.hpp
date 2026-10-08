#pragma once

// #include <cute/tensor.hpp>
#include <cute/algorithm/tuple_algorithms.hpp>
#include <cute/int_tuple.hpp>

// bring in the CUTE_HOST_DEVICE macro to allow
#include <cute/config.hpp>

#include <cstdint>
#include <iostream>

CUTE_HOST_DEVICE void pre_integer_sequence() {
  using namespace cute;
  // While integer sequence is not tuple, it exibit
  // tuple like property. So cute provides tuple_size, tuple_element
  // and get method for integer sequence, making it usable for places
  // where a tuple type is expcted.
  auto s = make_seq<5>{};
  print(tuple_size<decltype(s)>::value);
  print("\n");
  print(get<3>(s));
  print("\n");

  print(s);
  print("\n");
}

// void pre_tuple_transform()
// This signature is host only, and causes the lambda to be host only
// Does this imply that Lambda will inherit the host/device marker
// of the parent function.
CUTE_HOST_DEVICE void pre_tuple_transform() {
  using namespace cute;
  // Some prelimiary before diving into int_tuple
  // The first one is tuple transform, which is
  // basically a map function applied to tuple

  auto t = make_tuple(C<1>{}, C<2>{}, C<100>{});
  print(t);
  print("\n");

  auto res = transform(t, [](auto e) { return e + 1; });
  print(res);
  print("\n");

  // since the first paraeter of transform has T const &, not a perfect
  // forwarding parameter. it can only accept lvalue reference/
  // Note: the above analysis is incorrect, codex tells:
  // A const lvalue reference can bind to both lvalues and rvalues.
  // The restriction you’re thinking of applies to a non-const lvalue reference.
  // So the following code correctly compiles.
  auto res1 = transform(make_tuple(1, 2, 3), [](auto e) { return e + 1; });
  print(res1);
  print("\n");
}

CUTE_HOST_DEVICE void int_tuple_construct() {
  // int tuple is the basic type in cute, which is
  // expressed as a union type for either a integral type,
  // or a tuple of int tuples.

  using namespace cute;
  // Cute defines most of int tuple at type level.

  // ((4,5), (1,4)), a column major matrix.
  auto static_int_tuple = tuple<tuple<C<4>, C<5>>, tuple<C<1>, C<4>>>{};
  print(static_int_tuple);
  print("\n");

  // ((4,5), (5,1)), a row major matrix.
  auto static_int_tuple1 = tuple<tuple<C<4>, C<5>>, tuple<C<5>, C<1>>>{};
  print(static_int_tuple1);
  print("\n");

  // Cute also provide dynamic int_tuple implementatio
  // There are 3 ways to construct a dynamic int tuple

  // 1.
  // This is a purely dynamic int tuple with 2 elements, with at most 1 value
  // taken from the array, the rest of the tuple elements will be filled with
  // the init value.
  // The resulting value is basically a tuple with 2 dynamic element
  auto dyn_int_tuple = make_int_tuple<2>(cute::array<int, 3>{5, 4, 3}, 1, 100);
  print(dyn_int_tuple);
  print("\n");

  // zst array is also supported, althrough its useless because
  // every element is basically the same
  auto dyn_int_tuple1 = make_int_tuple<2>(cute::array<C<2>, 3>{}, 1, C<2>{});
  print(dyn_int_tuple1);
  print("\n");

  // The rest of the two methods are basically the same, they all create partial
  // dynamic int tuples The resulting int tuple has 3 dynamic values
  using ttype = tuple<tuple<int, int>, tuple<C<2>, int>>;
  auto mixed_dyn_tuple = make_int_tuple_from<ttype>(2, 4, 23);
  print(mixed_dyn_tuple);
  print("\n");

  // The other method need you to explictly construct an empty tuple,
  // and then letting that this method to replace the dynamic value stored
  // in the array to the corresponding position in the tuple
  auto mixed_dyn_tuple1 = ttype{};
  fill_int_tuple_from(mixed_dyn_tuple1, make_tuple(4, 5, 11));
  print(mixed_dyn_tuple1);
  print("\n");
}

CUTE_HOST_DEVICE void int_tuple_basic_op() {
  // We check basic int tuple operawtions here.
  using namespace cute;
  using int_tuple_ty = tuple<tuple<Int<2>, Int<3>>, tuple<int, int>>;
  auto t = make_int_tuple_from<int_tuple_ty>(5, 7);
  print(t);
  print("\n");

  // The get method is extended to the int tuple.
  // It matches each index from left to right and recursively
  // index into the target.
  // Note if an index is provided for a single integral value,
  // then that index must be 0.
  auto v1 = get<0, 1>(t);
  static_assert(v1 == Int<3>{});
  print(v1);
  print("\n");

  auto v2 = get<0, 1, 0>(t);
  static_assert(v2 == 3);
  print(v2);
  print("\n");

  auto v3 = get<1, 0>(t);
  print(v3);
  print("\n");

  auto v4 = get<1, 0, 0>(t);
  print(v4);
  print("\n");

  // rank
  // The api are similar, we can directly call rank,
  // we can also index into the tuple and call rank.
  print(rank(t));
  print("\n");

  print(rank<1>(t));
  print("\n");

  print(rank<1, 0>(t));
  print("\n");

  // shape
  // similar to shape
  // Also for single integeral value, we can index it infinitely with 0
  auto s = shape(t);
  print(s);
  print("\n");

  print(shape<1>(s));
  print("\n");

  print(shape<1, 0>(s));
  print("\n");

  print(shape<1, 0, 0>(s));
  print("\n");

  print(shape<1, 0, 0, 0, 0, 0, 0>(s));
  print("\n");

  // new finding:
  // max, min, gcd perform the reduce to the left
  // operation

  // max, finding the max value from the int tuple
  // From left to right, it calculate the max value of each
  // tuple element one by one.
  // For a single tuple element, if it is a tuple, it will be expanded
  // and recursively apply max again.
  // If it is a regular value, we call regular max function, and apply
  // the max to the rest of the tuple element.
  print(max(t));
  print("\n");

  print(max(t, 100, 7));
  print("\n");

  // min is similar to max
  print("%d\n", min(t, 0, 1));

  // GCD, similar,
  tuple<C<12>, C<16>> t1{};
  static_assert(gcd(t1) == C<4>{});
  print("%d\n", gcd(t1)());

  print("%d\n", gcd(tuple<tuple<C<8>, C<16>>, C<12>>{})());
  print("%d\n", gcd(tuple<C<8>, C<16>, C<12>>{})());

  // Depth
  // Each wrapped tuple increase the depth by 1
  print("%d\n", depth(tuple<tuple<C<8>, C<16>>, C<12>>{})());
  print("%d\n", depth(tuple<C<8>, C<16>, C<12>>{})());
  print("%d\n", depth<0, 1>(tuple<tuple<C<8>, C<16>>, C<12>>{})());

  // Product
  // Calculate the product of all the leaf values of the int tuple
  // Note: I almost forget that the pycute refer to each integral values
  // of the int tuple as leaf value.
  tuple<tuple<C<8>, C<16>>, tuple<C<3>, C<4>>, C<1>> t_p{};
  print("%d\n", Product{}(t_p)());


  // product each
  // calculate the product of each tuple element.
  // Return a new tuple of the product result.
  print(product_each(t_p));
  print("\n");

  print(product_each(1));
  print("\n");

  // product_like, do tuple product according to a 
  // provided tuple shape
  // Weakly congruent: guide ~ tuple
  tuple<tuple<C<8>, C<16>, tuple<C<2>, C<3>>>, tuple<C<3>, C<4>>, C<1>> base{};
  tuple<int, int, int> template1{};
  tuple<tuple<int, int, int>, int, int> template2{};
  tuple<tuple<int, int, tuple<int, int>>, int, int> template3{};
  auto res1 = product_like(base, template1);
  print(res1);
  print("\n");
  auto res2 = product_like(base, template2);
  print(res2);
  print("\n");
  auto res3 = product_like(base, template2);
  print(res2);
  print("\n");
  
  // size, the size of the int tuple, which is the multiplication
  // result of all the tuple leafs.
  print("%d\n", size(base)());

  // sum function, sum up all the tuple elements
  print("%d\n", sum(base)());

  // inner product, (a, b) inner_product (c, d) = a*b + c*d
  tuple<C<3>, C<4>> left_side{};
  auto righ_side = make_int_tuple_from<tuple<int, int>>(5, 6);
  print("%d\n", inner_product(left_side, righ_side));

  // Now let's see the behavior of ceil_div.
  // I think it exihibits different behavior 
  // depending on the size of the input tuples.
  
  // c1: if lhs's tuple size is larger than rhs's tuple size
  // then rhs will be filled with ones to match the size of the rhs
  // and then perform a pari-wise ceil_div for each tuple leaf.
  tuple<tuple<C<100>, C<50>>, C<4>, C<8>> a{};
  tuple<tuple<C<48>, C<71>>, C<2>> b{};  
  auto result = ceil_div(a, b);  
  print(result);
  print("\n");

  // c2: if rhs is an integer and lhs is a tuple, then it will be
  // calculated as rhs / product of lhs
  auto ceil_div_result2 = ceil_div(17, tuple<C<2>, C<4>>{});
  print(ceil_div_result2);
  print("\n");

  // c3: Each tuple element of lhs div with the rhs. The rhs is 
  // constantly changing, changing to rhs / tuple element. 
  auto ceil_div_result3 = ceil_div(tuple<C<6>, C<4>>{}, C<3>{});
  print(ceil_div_result3);
  print("\n");

  // c3.1: if the first leaf of lhs is larger than rhs. then only
  // the first leaf is divided, the rest of the tuple leafs are 
  // untouched. And the result preseves the shape lf lhs
  auto res_3_1_1 = ceil_div(tuple<C<6>, C<4>, tuple<C<7>, C<8>>>{}, C<3>{});
  print(res_3_1_1);
  print("\n");

  // c3.2: if the first leaf of lhs is smaller than rhs, then the leafs whose
  // product is larger or equal than the rhs will be replaced with 1, the rhs
  // is updated, and the process repeats
  auto res_3_1_2 = ceil_div(tuple<tuple<C<7>, C<8>>, C<6>, C<4>>{}, C<57>{});
  print(res_3_1_2);
  print("\n");
  
  // If one understands ceil_div, it is natural to understand shape_div
  // shape_div is similar to ceil_div, except that it requires 
  // lhs % rhs == 0 || rhs % lhs == 0
  // This won't compile, and report 
  // auto sd_1 = shape_div(tuple<C<6>, C<4>>{}, tuple<C<5>, C<3>>{});
  
  // congruent shapes are divided element-wise
  auto sd_2 = shape_div(
    tuple<C<6>, C<4>>{}, 
    tuple<C<3>, C<2>>{}
  );
  print(sd_2);
  print("\n");

  // weakly congruent shapes are divided with
  // the product of the rhs
  auto sd_3 = shape_div(
    tuple<C<6>, C<4>>{}, 
    tuple<tuple<C<3>, C<1>>, tuple<C<2>, C<2>>>{}
  );
  print(sd_3);
  print("\n");

  auto sd_4  = shape_div(    
    tuple<tuple<C<12>, C<6>>, tuple<C<24>, C<8>>>{},
    tuple<C<6>, C<4>>{}
  );
  print(sd_4);
  print("\n");

  // It basically tells you how to divide a shape
  // with a constant
  auto sd_5  = shape_div(    
    tuple<C<2>, C<4>, C<10>>{},
    C<40>{}
  );
  print(sd_5);
  print("\n");

}

void init_fuck() { std::cout << "fuck\n"; }