#include "cute/util/type_traits.hpp"
#include "numeric.hpp"
#include <cute/tensor.hpp>

#include <cstdint>
#include <iostream>
#include <type_traits>

namespace empty_base_exp {

// Define the generic template empty class here.
// The problem with this generic definition is that the generic
// bool variables are not tied with the generic type parameter T.
// So we need to establish a way between them
template <bool is_first_empty, bool is_rest_empty, class... T>
struct empty_base {};

// The way that we establish the connection is by defining some generic
// constexpr evaluated compile time. Which actually generate a constexpr
// boolean value based on the type of the parameters.
template <class First, class... Rest>
static constexpr bool is_first_empty = std::is_empty<First>::value;

template <class First, class... Rest>
static constexpr bool is_second_empty = (std::is_empty<Rest>::value && ...);

// Then we shrink the generic parametesr to only type parameters.
// Use the previously defined constexpr variable to partially instantiate
// empty_base. Then we establish a connection bewteen the template boolean
// variable with the template class type.
template <class... T>
using EB_t =
    struct empty_base<is_first_empty<T...>, is_second_empty<T...>, T...>;

// Use template partial instantiation to define concrete empty_base.
// What is partial instantiation:
// 1. Preserve some generic type parameters through template, like template
// template<class First, class... Rest>.
// 2. Fill in partial template parameters after the type definition: struct
// empty_base<false, false, First, Rest...>
template <class First, class... Rest>
// If First and Rest are all empty, then the struct is empty.
struct empty_base<true, true, First, Rest...> {};

template <class First, class... Rest>
struct empty_base<false, true, First, Rest...> {
  First first_;

  inline constexpr empty_base() : first_{} {};

  inline constexpr empty_base(First const &first, Rest const &...)
      : first_{first} {};
};

template <class First, class... Rest>
struct empty_base<true, false, First, Rest...> {
  EB_t<Rest...> rest_;

  inline constexpr empty_base() : rest_{} {};

  inline constexpr empty_base(First const &, Rest const &...rest)
      : rest_{rest...} {};
};

template <class First, class... Rest>
struct empty_base<false, false, First, Rest...> {
  EB_t<Rest...> rest_;
  First first_;

  inline constexpr empty_base() : first_{}, rest_{} {};

  inline constexpr empty_base(First const &first, Rest const &...rest)
      : first_{first}, rest_{rest...} {};
};

} // namespace empty_base_exp

namespace my_own {

namespace detail {
template <class... Ts>
constexpr uint64_t parse_int(uint64_t result, uint64_t first, Ts... rest) {
  if constexpr (sizeof...(Ts) == 0) {
    return 10 * result + first;
  } else {
    return parse_int(10 * result + first, rest...);
  }
}

} // namespace detail

template <char... Digits>
std::integral_constant<uint64_t, detail::parse_int(0, (Digits - '0')...)>
operator""_dddd() {
  return {};
}

} // namespace my_own
void test_integral_constant() {
  using namespace cute;

  //   // A constant value: short name and type-deduction for fast compilation
  // template <auto v>
  // struct C {
  //   using type = C<v>;
  //   static constexpr auto value = v;
  //   using value_type = decltype(v);
  //   CUTE_HOST_DEVICE constexpr operator   value_type() const noexcept {
  //   return value; } CUTE_HOST_DEVICE constexpr value_type operator()() const
  //   noexcept { return value; }
  // };
  // So first, cute defines compile-time constant.
  C<3> a{};
  C<4> b{};

  // Both these variables are empty variables
  static_assert(std::is_empty<C<3>>::value);
  static_assert(std::is_empty<C<4>>::value);

  int i = a;
  // The overload defined in C explicitly change the type conversion
  // and function behavior of all the variables defined with C<N>.
  std::cout << i << " " << (C<3>::value_type)a << " " << a() << "\n";

  // Algorithmic operations

  // Cute defines templates for some non-member operator overloading
  // which changes the operator behaviors for +-*/, etc.
  //   template<auto a, auto b>
  //   constexpr C<a + b> operator+(C<a>, C<b>) {
  //      return {}
  //   }
  auto sum = a + b;
  // at any moment, we can recover this abstract type
  // (computed at compile time but needed at runtime with)
  // decltype.
  std::cout << decltype(sum)::value << "\n";
  std::cout << sum << " expecting _7 \n";

  // Static dynamic mixinsg style
  // 1. Static mixes dynamic integers
  // Since each static integer overloads type conversion.
  // When being used to calculate with a dynamic integer,
  // it will be automatically converted to the corresponding dynamic
  // integer type by providing a corresponding dynamic integer.
  uint8_t z_ = 5;
  auto sum_new = z_ * C<3>{};
  std::cout << sum_new << " expecting 15 \n";
  std::cout << sizeof(decltype(sum_new)) << " expecting 4 \n";

  // 2. static static operations, using the provided overload.

  // 3. Special case static dynamic mixing
  // cute overload special static integers like C<0> for multiply.
  auto mul_new = 5 * C<0>{};
  std::cout << mul_new << " expecting _0 \n";
  std::cout << sizeof(decltype(mul_new)) << " expecting 1 \n";

  // Special overload from math.h
  // cute/numeric/math.hpp provides many math funcitons like abs
  // cute also provide static overload when executing these functions
  auto result = abs(-5);
  std::cout << result << " expecint 5 \n";
  auto static_result = abs(C<-5>{});
  std::cout << static_result << " expect _5 \n ";

  auto gcd_ = gcd(C<16>{}, C<12>{});
  std::cout << gcd_ << " expecting _4\n";

  // The conditional_return, which can be super useful for GPU programming.
  auto res = conditional_return(C<true>{}, C<3>{}, C<4>{});
  std::cout << res << " expecting _3\n";

  // Finally, cute also provide a quick shortcut for defining constant
  // This actually generates a compile-time constant constant<int, 54> C<54>
  auto v = 54_c;
  std::cout << v << " expecting _54\n";

  // This is so interesting that I exercise this function definition above
  // using std::integral_constant
  auto fres = my_own::detail::parse_int(0, 5, 4, 9);
  std::cout << fres << "\n";

  using namespace my_own;
  auto num_from_dddd = 3434_dddd;
  std::cout << decltype(num_from_dddd)::value << " expecting 3434 \n";

  std::cout << "end"
            << "\n";
}

void test_int_length() {
  using namespace cute;
  int_bit_t<2> x(1);

  // automatically int_bit_t dispatch
  // implicit conversion to one of the constructor
  // which checks whether the bit value is within its
  // required uppper bound
  // int_bit_t<2> x(2);
}

// An example seq definition
template <class T, T... Ints> struct Seq {
  using value_type = T;

  static constexpr size_t size() { return sizeof...(Ints); }
};

void test_integer_sequence() {
  // cute transform the std::integer_sequence into its own
  // integer_sequence type. It only takes the template parameters
  // of the std::integer_sequence, and then pass these template parameters
  // to the cute::integer_sequence.
  using S5 = cute::make_integer_sequence<int, 5>;
  std::cout << S5::size() << " \n";

  using seq_1_5 = cute::make_range<1, 5>;
  std::cout << seq_1_5::size() << " \n";

  // Basically everything here is a sequence at the type level.
  // Seq shows a definition example.
  
  // What makes me feel most interesting is this part/
  // We can define a tuple with compile time integers.
  auto t = cute::make_tuple(cute::C<1>{}, cute::C<2>{}, cute::C<3>{}, cute::C<411>{});
  // Then we can convert it to a seq
  using t_seq = cute::to_seq<decltype(t)>::type;
  // t_seq has type: seq<1,2,3>.
  std::cout<< t_seq::size() <<" \n";

  // The final part, tuple_size and tuple_element has a partial specialization for 
  // integer_sequence. Therefore we can retrieve the corresponding index and query the 
  // actual size of the integer sequence
  std::cout<<cute::tuple_size<t_seq>::value <<" expecting 4\n";
  std::cout<<cute::tuple_element<2, t_seq>::type::value << " expecting 3\n";

  auto const_val = cute::get<3>(t_seq{});
  // The type conversion overload
  std::cout<<const_val+1<<" expecting 412\n";
  // The function call overload
  std::cout<<const_val()<<" expecting 411\n";
  // The iostream overload
  std::cout<<const_val<<" expecting _411\n";
}