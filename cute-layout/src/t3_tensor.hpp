#pragma once

#include "cute/config.hpp"
#include "cute/pointer.hpp"
#include "cute/stride.hpp"
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

void test_tensor() {
  using namespace cute;
  print("fuck\n");

  // First, let's test how under_score
  // selection is used.
  Layout<Shape<Shape<_3, _4>, Shape<_2, _3, _2>>,
         Stride<Stride<_1, _3>, Stride<_2, _3, _4>>>
      a;
  print(a);
  print("\n");

  // auto a1 = make_tuple(2, 3);
  // print("%d\n", tuple_size<decltype(a)>::value);
  // print("%d\n", tuple_size<Layout<Shape<Shape<_3, _4>, _2, _3, _2>,
  // Stride<Stride<_1, _3>, _2, _3 ,_4>>>::value);

  print(a(_, make_tuple(_, 2, _)));
  print("\n");
}

void test_layout_slice() {
  using namespace cute;
  // Let's make a difference between slicing a
  // layout verses slicing a tensor which is going
  // to discuss later.

  // By slicing a layout, we are only extracting the interesting
  // modes that matches the _. Non-underscore tuple values will
  // be ignored.

  Layout<Shape<Shape<_3, _4>, Shape<_2, _3, _2>>,
         Stride<Stride<_1, _3>, Stride<_2, _3, _4>>>
      a;

  // (3, 4) only
  print(a(_, 1));
  print("\n");

  // (3,4),(2,3)
  print(a(_, make_tuple(_, _, 1)));
  print("\n");

  // 4, 3
  print(a(make_tuple(1, _), make_tuple(2, _, 1)));
  print("\n");
}

__global__ void run_testor(float *gA, int dynamic_shape) {
  using namespace cute;

  auto tensor_8 = make_tensor(gA, make_layout(Int<8>{}));
  print(tensor_8);
  print("\n");

  auto tensor_8s = make_tensor(gA, Int<8>{});
  print(tensor_8s);
  print("\n");

  auto tensor_8d2 = make_tensor(gA, 8, 2);
  print(tensor_8d2);
  print("\n");

  auto gmem_8s = make_tensor(make_gmem_ptr(gA), Int<8>{});
  print(gmem_8s);
  print("\n");

  auto gmem_8d = make_tensor(make_gmem_ptr(gA), 8);
  print(gmem_8d);
  print("\n");

  auto gmem_8sx16d =
      make_tensor(make_gmem_ptr(gA), make_shape(Int<8>{}, dynamic_shape));
  print(gmem_8sx16d);
  print("\n");

  auto gem_8dx16s = make_tensor(make_gmem_ptr(gA), make_shape(8, Int<16>{}),
                                make_stride(Int<16>{}, _1{}));
  print(gem_8dx16s);
  print("\n");

  Layout<Shape<_4, _8>> smem_layout{};
  constexpr auto smem_size = decltype(cosize(smem_layout))::value;
  print(smem_size);
  print("\n");
  __shared__ float smem[smem_size];

  auto smem_4x8_col = make_tensor(make_smem_ptr(smem), smem_layout);
  print(smem_4x8_col);
  print("\n");

  auto smem_4x8_row = make_tensor(make_smem_ptr(smem), shape(smem_layout),
                                  compact_row_major(shape(smem_layout)));
  print(smem_4x8_row);
  print("\n");

  auto rmem_4x8_col = make_tensor<float>(Shape<_4, _8>{});
  auto rmem_4x8_row = make_tensor<float>(Shape<_4, _8>{}, LayoutRight{});
  auto rmem_4x8_pad = make_tensor<float>(Shape<_4, _8>{}, Stride<_32, _2>{});
  auto rmem_4x8_like = make_tensor_like(rmem_4x8_pad);

  print(rmem_4x8_col);
  print("\n");
  print(rmem_4x8_row);
  print("\n");
  print(rmem_4x8_pad);
  print("\n");
  print(rmem_4x8_like);
  print("\n");

  {
    auto A = make_tensor<float>(Shape<Shape<_4, _5>, Int<13>>{},
                                Stride<Stride<Int<12>, _1>, _64>{});

    auto B = make_tensor(make_gmem_ptr(gA), make_shape(13, 20));

    print(A);
    print("\n");
    print(B);
    print("\n");

    for (int m0 = 0; m0 < size<0, 0>(A); m0++) {
      for (int m1 = 0; m1 < size<0, 1>(A); m1++) {
        for (int n = 0; n < size<1>(A); n++) {
          A[make_coord(make_coord(m0, m1), n)] = n + 2 * m0;
        }
      }
    }

    // Do a transpose
    for (int m = 0; m < size<0>(A); m++) {
      for (int n = 0; n < size<1>(A); n++) {
        B(n, m) = A(m, n);
      }
    }

    // Copy back, but what is the resulting tensor?
    for (int i = 0; i < size(A); i++) {
      A[i] = B[i];
    }
  }

  {
    // Here, let's examine how to slice a tensor
    auto l = make_layout(
        make_shape(make_shape(Int<3>{}, 2), make_shape(2, Int<5>{}, Int<2>{})),
        make_stride(make_stride(4, 1), make_stride(Int<2>{}, 13, 100)));

    print("%d\n", crd2idx(make_coord(2, _), l));
    print("%d\n", crd2idx(make_coord(2, 0), l));

    print("%d\n", crd2idx(make_coord(_, 5), l));
    print("%d\n", crd2idx(make_coord(0, 5), l));

    print("%d\n", crd2idx(make_coord(make_coord(_, _), 5), l));
    print("%d\n", crd2idx(make_coord(make_coord(0, 0), 5), l));

    print("%d\n", crd2idx(make_coord(make_coord(_,1),make_coord(0, _, 1)), l));
    print("%d\n", crd2idx(make_coord(make_coord(0,1), make_coord(0, 0, 1)), l));

    print("%d\n", crd2idx(make_coord(make_coord(2,_),make_coord(_, 3, _)), l));
    print("%d\n", crd2idx(make_coord(make_coord(2,0), make_coord(0, 3, 0)), l));
  }

  {
    auto a = make_tensor(gA, make_shape(8, 24));
    auto tiler = Shape<_4, _8>{};

    auto tiled_a = zipped_divide(a, tiler);
    print(tiled_a);
    print("\n");

    print("%d, %d, %d\n", blockIdx.x, blockIdx.y, blockIdx.z);
    auto cta_a = tiled_a(_, make_coord(0, 0));

    print(cta_a);
    print("\n");

    print(tiled_a(_, make_coord(1, 0)));
    print("\n");

  }
}