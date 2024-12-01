#ifndef IVP_SSPRK3_HPP
#define IVP_SSPRK3_HPP

#include "../numcepts/numcepts.hpp"
#include <vector>

namespace ivp
{
  /**
   * @brief Implements the three stage third order stability preserving Runge
   * Kutta method.
   *
   * @tparam Y scalar or vector type for y-variable. float, complex<...>,
   * vector<...>, array<...>, etc.
   */
  template <typename Y>
  class SSPRK3;

  // specialization of SSPRK3 for scalar types.
  template <numcepts::ScalarType Y>
  class SSPRK3<Y>
  {
  public:
    using real = numcepts::precision_t<Y>;

    SSPRK3() = default;

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
      Y u = y + dt * f(t, y);

      real s = t + dt;
      real A = real(0.75), B = real(0.25), C = real(0.25) * dt;

      u = A * y + B * u + C * f(s, u);

      s = t + real(0.5) * dt;
      A = real(1.0 / 3.0), B = real(2.0 / 3.0), C = real(2.0 / 3.0) * dt;

      y = A * y + B * u + C * f(s, u);
      t += dt;
    }
  };

  // specialization of SSPRK3 for vector types.
  template <std::ranges::forward_range Y>
  class SSPRK3<Y>
  {
  public:
    using real = numcepts::precision_t<Y>;

    template <typename... Args>
    SSPRK3(Args &&...args) : p(args...), u(args...) {}

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

      auto yi = std::begin(y);
      auto pi = std::begin(p);
      for (auto ui = std::begin(u); ui != std::end(u); ++yi, ++pi, ++ui)
        (*ui) = (*yi) + dt * (*pi);

      real s = t + dt;
      real A = real(0.75), B = real(0.25), C = real(0.25) * dt;
      f(p, s, u);

      yi = std::begin(y);
      pi = std::begin(p);
      for (auto ui = std::begin(u); ui != std::end(u); ++yi, ++pi, ++ui)
        (*ui) = A * (*yi) + B * (*ui) + C * (*pi);

      s = t + real(0.5) * dt;
      A = real(1.0 / 3.0), B = real(2.0 / 3.0), C = real(2.0 / 3.0) * dt;
      f(p, s, u);

      yi = std::begin(y);
      pi = std::begin(p);
      for (auto ui = std::begin(u); ui != std::end(u); ++yi, ++pi, ++ui)
        (*yi) = A * (*yi) + B * (*ui) + C * (*pi);

      t += dt;
    }

  private:
    mutable Y p;
    mutable Y u;
  };

  // specialization of SSPRK3 for arrays (pointers) of scalar types.
  template <numcepts::ScalarType Y>
  class SSPRK3<Y *>
  {
  public:
    using real = numcepts::precision_t<Y>;

    SSPRK3(size_t dim) : n{dim}, p(dim), u(dim) {}

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
        u[i] = y[i] + dt * p[i];

      real s = t + dt;
      real A = real(0.75), B = real(0.25), C = real(0.25) * dt;
      f(p.data(), s, u.data());

      for (size_t i = 0; i < n; ++i)
        u[i] = A * y[i] + B * u[i] + C * p[i];

      s = t + real(0.5) * dt;
      A = real(1.0 / 3.0), B = real(2.0 / 3.0), C = real(2.0 / 3.0) * dt;
      f(p.data(), s, u.data());

      for (size_t i = 0; i < n; ++i)
        y[i] = A * y[i] + B * u[i] + C * p[i];

      t += dt;
    }

  private:
    const size_t n;
    mutable std::vector<Y> p;
    mutable std::vector<Y> u;
  };

} // namespace ivp

#endif
