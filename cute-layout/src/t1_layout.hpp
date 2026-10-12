#pragma once

#include "cute/stride.hpp"
#include <cute/layout.hpp>

void fuck() {
  using namespace cute;
  
  auto s1 = make_shape(Int<2>{}, C<3>{});
  print(s1);
  print("\n");

  auto stride1 = make_stride(C<1>{}, C<2>{});
  print(stride1);
  print("\n");
}

void test_make_layout() {
  using namespace cute;

  // Layout is basically a rank-2 tuple, with the first being the shape
  // and the second being the stride.
  auto l1 = make_layout(make_shape(3,4), make_stride(4, 5));
  print(l1);
  print("\n");

  // the shape and the stride can be accessed
  // basic accessors are provided
  print(l1.shape());
  print("\n");

  print(l1.stride());
  print("\n");

  // Static layout can be constructed with ease.
  Layout<Shape<Int<4>, Int<5>>, Stride<Int<1>, Int<4>>> l2{};
  print(l2);
  print("\n");
  print(l2.stride());
  print("\n");
  print(l2.shape());
  print("\n");

  // layout with mixed static and dynamic values can be constructed.
  auto l3 = make_layout(make_shape(3, Int<4>{}), make_stride(Int<4>{}, 1));
  print(l3);
  print("\n");
  print(l3.shape());
  print("\n");
  print(l3.stride());
  print("\n");

  // The default behavior of make_layout
  // cute can automatically infer the stride. The resulting
  // layout will be a column-major vector
  auto l4 = make_layout(make_shape(3, Int<4>{}));
  print(l4);
  print("\n");

  auto l5 = make_layout(make_shape(3));
  print(l5);
  print("\n");

  auto l6 = make_layout(make_shape(3, 1, 3));
  print(l6);
  print("\n");

  // If the shape has a static 1, then the corresponding 
  // stride will be set to 0. 
  // But a dynamic 1 behaves the same as other strides values.
  // see previous l6 example.
  // In theory we can set any stride value for the 1 in the shape. 
  // But why does cute choose this setup.
  auto l7 = make_layout(make_shape(3, Int<1>{}, Int<3>{}));
  print(l7);
  print("\n");
}

void test_compact_major() {
  using namespace cute;

  // compact_major, it's hard to tell what it does just from its name,
  // so we use some concrete examples for explain it.
  // compact_major<LayoutLeft> returns a col-major stride for a 2d shape.
  // compact_major<LayoutRight> returns a row-major stride instead. 

  // Considering what a col-major matrix looks like, its elements are placed
  // compctly along each matrix column, whereas the row-major matrix's row elements
  // are placed compactly within the continguous memory space.

  // Let's expand this basic scenario to shapes with arbitrary ranks. 
  // compact_major<LayoutLeft> returns a stride where the elements are compacted
  // placed in the contiguous memory space from the first mode to the last. 
  // Therefore, the elements will look like this in the memory:
  // (0, 0, 0, 0, xx), (1, 0, 0, 0, 0xx), (2, 0, 0, 0, 0, xx);
  // If we convert the shape cordinate to index, then we can see that 
  // the indexes increase monotonically. 
  
  // On contrary, as its variable type tag suggests, compact_major<LayoutRight> returns a 
  // stride, causing elements on the right-most mode to be placed contiguously in the memory.
  // (xxx,0,0,1), (xxx,0,0,2), (xxxx,0,0,3).

  // How the compact major is implemented using C++ template metaprogramming:
  // it's similar to ceil_div, you create a initial context with an empty tuple and
  // an integer value that you are going to use for the computation. 
  // Then you fold each tuple element, and recursively call the compact_major if the 
  // element is an int tuple againt. For a regualr integer value, you just 
  // compute a new result (current_value * tuple value), insert the old current_value
  // to the tuple stored in the fold state. Replace the old current_value with the new 
  // computed result, and continue until the last tuple element. 
  // Then return the result and take the first element from the resulted tuple. 

  auto r1 = compact_major<LayoutLeft>(make_tuple(2, 3, 4));
  print(r1);
  print("\n");

  auto r2 = compact_major<LayoutLeft>(make_tuple(make_tuple(3,5), make_tuple(4,2), 4));
  print(r2);
  print("\n");

  auto r3 = compact_major<LayoutLeft>(make_tuple(5, 6));
  print(r3);
  print(" we quickly get the col-major stride for (5, 6)\n");

  auto r4 = compact_major<LayoutRight>(make_tuple(5, 6));
  print(r4);
  print(" this is the col-major stride for (5, 6)\n");

  // For this, it bassically changes the basic stride to
  // 4, therefore we will have a stride of (4, 20) as the result.
  auto r5 = compact_major<LayoutLeft>(make_tuple(5,6), 4);
  print(r5);
  print("\n");

  // If the base stride is a tuple, then it indicates that the 
  // two subtuples of the input shape will use each value in the 
  // stride tuple as the base stride.
  auto r6 = compact_major<LayoutLeft>(make_tuple(make_tuple(2,3), make_tuple(3,4)), make_tuple(1, 2));
  print(r6);
  print("\n");
}

void test_idx_crd_conversion() {
  using namespace cute;
  // First, let's examine crd2idx
  // Given a coordinate, like (0, 2) for a shpae of (3, 4),
  // It's like you passing a list of integers (2, 0), but the 
  // base of each integer is (4, 3). 
  // Then you just need to convert the list of integers to 
  // its actual decimal value. 
  // The method is straightforward just as we are calculating 
  // a list of integers (1,2,3,4), with base being (10, 10, 10, 1).
  // You first calculate the multipliers for each integer, which 
  // are (1000, 100, 10, 1). Then you sum each multipiler with 
  // the integer and add everything up (1*1000+2*100+3*10+4*1).
  // So, for crd2idx, we can first calculate the compact left stride
  // of the shape, which serves as the base multiplers for each cordinate vlaue.
  // Then we sum things together. 
  // Cute uses another algorithm that suits recursive template metaprogramming.
  // c0 + s0*(c1 + s1* (c2 + s2*....)).
  // If you expands this formular, you get exactly the same thing. 
  print(crd2idx(make_tuple(2,4), make_tuple(5, 8)));// 2*1 + 4*5.
  print("\n");

  //The provided cord should be weakly congruent with the shape.
  // In that case, each individual cord value describes a postiion 
  // at the prod(s) space.
  print(crd2idx(make_tuple(2,4), make_tuple(make_tuple(2,3), make_tuple(2,4))));
  // equipvalent to cord: (2,4) shape: (6, 8) 2*1+4*6
  print("\n");


  // crd2idx can be computed together with a stride.
  // In that case, it returns the final idx after the stride
  auto shape1 = make_shape(3,5);
  auto stride1 = make_stride(4, 6);
  print(crd2idx(make_tuple(2, 3), shape1, stride1));
  // For the most simple casee, we have 2*4+3*6.
  print("\n");

  // If the provided coord is an integer, then
  // crd2idx first treats it as an index and finds out the 
  // corresponding coordinates, and then comput using the 
  // regular rule. Actually, idx2crd reuses the computation
  // logic in of crd2idx.
  // a 3D shape
  auto shape2 = make_shape(3,6,7);
  auto stride2 = make_stride(2,8,9);
  print(crd2idx(51, shape2, stride2));
  // first, convert the 51, which is the idx, to the real 3d coord
  // 51 % 3, (51/3)%6, (51/18)%7
  // 0         5          2
  // The result will be 0+40+18 = 58
  // then calculate the resulting index with stride2
  print("\n");

  // idx2crd
  // Treat idx as decimal number and obtain each integer on the 
  // corresponding coordinate position 
  // For regular version, we can use 
  // idx % s0
  // (idx / s0) % s1
  // (idx/ (s0*s1)) % s2
  // ... 
  // There are multiple ways for implement idx2crd.
  // One way is to use fold, which constructs the resulting 
  // coord (congruent with the shape) as an intemediate state,
  // which also contains the division result used to calculate
  // each seperate coordinate. 
  // cute chooses to reuse crd2idx. 
  // as crd2idx already computes the coordinate result if the provided
  // coordinate is a simple integer. 
  // It uses base vector as the stride. 
  // Therefore, for a shape of (2,3), the stride will be (E<0>, E<1>).
  // for a cordinate of (1,1), the result will be 
  // E<0> + E<1>, which are convert to tuple add
  // (1,0) + (0, 1) -> (1, 1), which is exactly the 
  // coordinate of index 3.
  print(idx2crd(51, shape2));
  // This will be (0,5,2), calculated from 0*E<0>+5*E<1>+2*E<2>
  print("\n");

  // idx2crd can also carry stride. In that case, it calculate the coordinate
  // using the shape-stride layout. 
  // What it does is also quite simple, try to recover the 
  // congruent shape-stride pair. Then obtain each coordinate
  // by divinding the stride first and then dividing the corresponding
  // shape for the remainder. 
  // Note crd2idx only works for compact layout. 
  // None compact layout has some problems. 
  // For instance: shape2 and stride2 form a pair of
  // non compact layout. So the result second result is not 
  // (2,4,5). I think this relates to the surjective/bijective
  // injective relationship talked throughout the cute paper.
  print(crd2idx(make_tuple(2,4,5), shape2, stride2));
  print("\n");
  print(idx2crd(81,shape2, stride2));
  print("\n");

  // For a compact shape, this becomes correct
  auto compact_stride = compact_col_major(shape2);
  print(crd2idx(make_tuple(2,4,5), shape2, compact_stride));
  print("\n");
  print(idx2crd(104,shape2, compact_stride));
  print("\n");


}