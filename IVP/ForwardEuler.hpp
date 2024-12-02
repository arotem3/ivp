#ifndef IVP_FORWARD_EULER_HPP
#define IVP_FORWARD_EULER_HPP

#include "numcepts/numcepts.hpp"
#include <vector>

namespace ivp
{
  /**
   * @brief Implements the explicit Euler method for solving the initial value
   * problem: y' = F(t, y).
   *
   * @tparam Y scalar or vector type for y-variable. float, complex<...>,
   * vector<...>, array<...>, etc.
   */
  template <typename Y>
  class ForwardEuler;

  // specialization of ForwardEuler for scalar types.
  template <numcepts::ScalarType Y>
  class ForwardEuler<Y>
  {
  public:
    using real = numcepts::precision_t<Y>;

    ForwardEuler() = default;

    /**
     * @brief computes a single step fo the IVP y' = F(t, y)
     *
     * @tparam Func invocable as f(real, const Y &)
     * @param dt time step
     * @param f invocable as f(t, y). Returns F(t, y).
     * @param t current t-value. On exit, t = t + dt.
     * @param y current y-value. On exit, y ~ y(t + dt).
     */
    template <typename Func>
    void step(real dt, Func &&f, real &t, Y &y) const
    {
      y += dt * f(t, y);
      t += dt;
    }
  };

  // specialization of ForwardEuler for vector types.
  template <std::ranges::forward_range Y>
  class ForwardEuler<Y>
  {
  public:
    using real = numcepts::precision_t<Y>;

    /**
     * @brief Construct a new Forward Euler object.
     *
     * @param args passed to Y constructor of all work variables. e.g. when
     * Y is a std::vector<...> then the size should be passed as an argument.
     */
    template <typename... Args>
    ForwardEuler(Args &&...args) : p(args...) {}

    /**
     * @brief computes a single step of the IVP y' = F(t, y)
     *
     * @tparam Func invocable as f(Y&, real, const Y &)
     * @param dt time step
     * @param f invocable as f(p, t, y). On exit, p = F(t, y)
     * @param t current t-value. On exit, t = t + dt.
     * @param y current y-value. On exit, y ~ y(t + dt).
     */
    template <typename Func>
    void step(real dt, Func &&f, real &t, Y &y) const
    {
      f(p, t, y);

      auto pi = std::begin(p);
      for (auto yi = std::begin(y); yi != std::end(y); ++yi, ++pi)
        (*yi) += dt * (*pi);

      t += dt;
    }

  private:
    mutable Y p;
  };

  // specialization of ForwardEuler for arrays (pointers) of scalar types.
  template <numcepts::ScalarType Y>
  class ForwardEuler<Y *>
  {
  public:
    using real = numcepts::precision_t<Y>;

    /**
     * @brief Construct a new Forward Euler object
     *
     * @param dim dimension of y.
     */
    inline explicit ForwardEuler(size_t dim) : n{dim}, p(dim) {}

    /**
     * @brief computes a single step of the IVP y' = F(t, y)
     *
     * @tparam Func invocable as f(Y *, real, const Y *)
     * @param dt time step
     * @param f invocable as f(p, t, y). On exit, p = F(t, y)
     * @param t current t-value. On exit, t = t + dt.
     * @param y current y-value. On exit, y ~ y(t + dt).
     */
    template <typename Func>
    void step(real dt, Func &&f, real &t, Y *y) const
    {
      f(p.data(), t, y);

      for (size_t i = 0; i < n; ++i)
        y[i] += dt * p[i];
    }

  private:
    const size_t n;
    mutable std::vector<Y> p;
  };

} // namespace ivp

#endif
