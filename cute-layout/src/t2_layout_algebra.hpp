#pragma once

#include <cute/layout.hpp>

void test_algebra() {
  using namespace cute;
  print(coalesce(make_layout(Int<1>{}, Int<0>{})));
  print("\n");

  print(coalesce(make_layout(Int<1>{}, Int<1>{})));
  print("\n");

  print(coalesce(make_layout(make_shape(Int<2>{}, Int<4>{}))));
  print("\n");

  print(coalesce(make_layout(make_shape(Int<2>{}, Int<4>{}, Int<6>{}))));
  print("\n");

  print(coalesce(make_layout(make_shape(Int<2>{}, Int<4>{}), GenRowMajor{})));
  print("\n");

  {
    auto l = make_layout(make_shape(_5{}, make_shape(_2{}, _3{})));
    auto fl = flatten(l);
    print(get<2>(shape(fl)));
    print("\n");
  }

  // dynamic values can not be coalesced. note this.
  // The first two modes will be coalesced because they
  // are static integers.
  print(coalesce(make_layout(make_shape(Int<2>{}, Int<4>{}, 6))));
  print("\n");

  // This will not coalesced becauses the strides of the last
  // two modes are all dynamic values and will not coalesced
  // during a backward coalescing process.
  print(coalesce(make_layout(make_shape(2, Int<4>{}, Int<6>{}))));
  print("\n");

  print(
      coalesce(make_layout(make_shape(_5{}, _4{}, _2{}, _1{}), GenRowMajor{})));
  print("\n");

  // By mode coalesce
  // The provided mode may be smaller than the rank size.
  // Remaining ranks will be ignored.
  auto a =
      Layout<Shape<_2, Shape<_1, _6>, _3>, Stride<_1, Stride<_6, _2>, _4>>{};
  print(coalesce(a, tuple<_1, _1>{}));
  print("\n");

  print(coalesce(a, tuple<_1>{}));
  print("\n");

  print(coalesce(a, make_tuple(_1{}, _)));
  print("\n");

  print(coalesce(a, make_tuple(_1{}, make_tuple(_))));
  print("\n");

  // print(coalesce_x(Layout<Shape<_4, _1>, Stride<_1, _0>>{}));
  // print("\n");

  print(coprofile(Layout<Shape<_4, Shape<_2, _3>>,
                         Stride<E<0>, Stride<E<1, 0>, E<1, 1>>>>{}));
  print("\n");

  auto rhs = Layout<Shape<_5, _4, _3, _1>>{};
  print(rhs);
  print("\n");
  // (5,4, 3, 2)
  // (1,5  20, 60)
  // auto rhs = Layout<Shape<_5, _4, _3, _1>, Stride<_3, _3, _2, _3>>{};
  auto lhs = Layout<_4, _5>{};
  auto res = detail::coalesce_x(rhs, coprofile(lhs));
  // auto res = coalesce(rhs, coprofile(lhs));
  print(res);
  print("\n");
}

void test_composition() {
  using namespace cute;
  Layout<Shape<_10, _5>, Stride<_5, _1>> l1;
  for (int i = 0; i < get<0>(shape(l1)); i++) {
    for (int j = 0; j < get<1>(shape(l1)); j++) {
      print("%d ", l1(i, j));
    }
    print("\n");
  }

  // Layout<Shape<_2, Int<25>>, Stride<_1, _2>> l2{};
  Layout<_2, _1> l2;
  auto compos_res = composition(l1, l2);
  print(compos_res);
  print("\n");

  {
    // a -> coalesce to (30, 1)
    Layout<Shape<_6, _5>, Stride<_1, _6>> a;
    Layout<_4, _3> b;
    // The result becomes (4, 3*1)
    auto res = composition(a, b);
    print(composition(a, b));
    print("\n");
  }

  {
    // Here we try to check some value conditions for
    // composition.
    // There are two major assertions for composition.
    // First, the stride value used for composing,
    // should either be smaller than the current shape,
    // or can fully divide the current shape. Otherwise
    // an error will be thrown.

    Layout<Shape<_6, _5>, Stride<_2, _6>> a;
    // This can not be compiled, because _7 % _6 != 0.
    // Layout<_4, _7> b;
    // auto res = composition(a, b);
    // print(composition(a, b));
    // print("\n");
  }

  {
    // After a more thorough investigation, I found the
    // compostiion to follow 3 steps.

    // Let's treat the following step as the first example
    Layout<Shape<_3, _2, _10>, Stride<_2, _4, _7>> a;
    Layout<_4, Int<30>> b;

    // There is a core condition for composition computation, which is
    // rest_stride % curr_shape) == Int<0>{}
    // Threefore, if the rest_stride is larger than the currange shape
    // It must be able to fully divided it (remainder being 0).
    // Also, this causes the next_shape to be 1, and it will directly
    // jump to the next fold iteration, and the only thing that's changed is the
    // rest_stride, which will now be _30 / _3 = _10.

    // Therefore, the first step in the compostiion computation, is that
    // if the stride is larger than the shape, then it will gradually divide
    // the shapes, and remove those shapes that are smaller than the
    // rest_stride.

    // This is reasonable, we can think of this process as something like this
    // We make one step in the inner layout, which will step out an initial
    // rest_stride It will pass all the shapes that can be fully divided. For
    // instance, in the above example, the initial step is 30. It will pass 3,
    // leaving 30/3 strides. It then passes 2, leaving 30/3/2 strides. Then it
    // will find its position within the last shape.

    // Therefore, the result will be (4, 35)
    print(composition(a, b));
    print("\n");
  }

  {
    Layout<Shape<_10, _4, _2>, Stride<_2, _4, _7>> a;
    Layout<_4, _3> b;

    // The second condition looks similar to the example above.
    // Just imagine that the first condition continues, until the
    // rest_stride is now smalelr than than the shape.
    // Then we can do regular computation.
    // Here, next_shape will be 10 ceil_div 3 = 4
    // Next stride can only be 1.

    // Here, the next_shape is larger or equal
    // to the rest_shape. It means that we have found
    // a mode where we can accomodate all the indexes
    // from the inner layout.
    // So it will produce a mode of (4,...),
    // with rest_stride and rest_shape all being 1.

    // Then the fold will just skill the rest of the modes
    // and produces under this example: (4, 6)
    print(composition(a, b));
    print("\n");
  }

  {
    print("fuck?\n");
    // The final example looks like this, the rest_stride
    // is smaller than a series of modes. Then it will gradually shrink
    // the size of the rest_shape, until 1 is reached. 
    // It fuilly divides an outer mode.
    // inner layout: 0, 5, 10, 15
    // Every 2 indexes step through the 10 mode of the outer layout. 
    Layout<Shape<_10, _4, _2>,  Stride<_2, _4, _7>> a;

    // Note, this does not compile, because
    // The first mode is (9/3=3), and can
    // not be fully divided by 4.
    // Layout<_4, _4> b;

    // Expecting a (2,2):(10, 4)
    Layout<_4, _5> b;

    print(composition(a, b));
    print("\n");
  }

  // Therefore, except for the last mode of the outer layout.
  // Every other mode in the outer layout must obey some fully 
  // devided rule. 
  // Either the rest_stride can fully divided these modes.
  // Or the rest_shape can fully divide the next_shape. 
  // This is the basic rule of the composition. 
  // Not any two layouts can be composed together, they must obey 
  // the layout composition rule. 
}