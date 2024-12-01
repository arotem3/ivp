#ifndef IVP_RK2_HPP
#define IVP_RK2_HPP

#include "../numcepts/numcepts.hpp"
#include <vector>

namespace ivp
{
  /**
   * @brief Implements the two stage second order Runge Kutta method, i.e. the
   * explicit midpoint method, for solving the initial value problem: y' = F(t, y).
   *
   * @tparam Y scalar or vector type for y-variable. float, complex<...>,
   * vector<...>, array<...>, etc.
   */
  template <typename Y>
  class RK2;

  // specialization of RK2 for scalar types.
  template <numcepts::ScalarType Y>
  class RK2<Y>
  {
  public:
    using real = numcepts::precision_t<Y>;

    RK2() = default;

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
      Y p = f(t, y);

      const real half_dt = real(0.5) * dt;
      Y u = y + half_dt * p;

      const real s = t + half_dt;
      p = f(s, u);

      y += dt * p;
      t += dt;
    }
  };

  // specialization of RK2 for vector types.
  template <std::ranges::forward_range Y>
  class RK2<Y>
  {
  public:
    using real = numcepts::precision_t<Y>;

    template <typename... Args>
    RK2(Args &&...args) : p(args...), u(args...) {}

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

      const real half_dt = real(0.5) * dt;

      auto pi = std::begin(p);
      auto ui = std::begin(u);
      for (auto yi = std::begin(y); yi != std::end(y); ++yi, ++ui, ++pi)
        (*ui) = (*yi) + half_dt * (*pi);

      const real s = t + half_dt;
      f(p, s, u);

      pi = std::begin(p);
      for (auto yi = std::begin(y); yi != std::end(y); ++yi, ++pi)
        (*yi) += dt * (*pi);

      t += dt;
    }

  private:
    mutable Y p;
    mutable Y u;
  };

  // specialization of RK2 for arrays (pointers) of scalar types.
  template <numcepts::ScalarType Y>
  class RK2<Y *>
  {
  public:
    using real = numcepts::precision_t<Y>;

    RK2(size_t dim) : n{dim}, p(dim), u(dim) {}

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

      const real half_dt = real(0.5) * dt;
      for (size_t i = 0; i < n; ++i)
        u[i] = y[i] + half_dt * p[i];

      const real s = t + half_dt;
      f(p.data(), s, u.data());

      for (size_t i = 0; i < n; ++i)
        y[i] += dt * p[i];

      t += dt;
    }

  private:
    const size_t n;
    mutable std::vector<Y> p;
    mutable std::vector<Y> u;
  };

} // namespace ivp

#endif
