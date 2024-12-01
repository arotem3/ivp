#ifndef IVP_LSRK4_HPP
#define IVP_LSRK4_HPP

#include "../numcepts/numcepts.hpp"
#include <vector>

namespace ivp
{
  /**
   * @brief Implements the explicit low storage five stage fourth order Runge
   * Kutta method for solving the initial value problem y'(t) = F(t, y).
   *
   * @tparam Y scalar or vector type for y-variable. float, complex<...>,
   * vector<...>, array<...>, etc.
   */
  template <typename Y>
  class LSRK4;

  // specialization of LSRK4 for scalar types.
  template <numcepts::ScalarType Y>
  class LSRK4<Y>
  {
  public:
    using real = numcepts::precision_t<Y>;

    LSRK4() = default;

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
      constexpr real rk4a[5] = {0.0, -0.41789047449985196221, -1.1921516946426769261, -1.6977846924715278362, -1.5141834442571557816};
      constexpr real rk4b[5] = {0.14965902199922911733, 0.37921031299962728091, 0.82295502938698171717, 0.69945045594912210704, 0.15305724796815199267};
      constexpr real rk4c[5] = {0.0, 0.14965902199922911733, 0.37040095736420477295, 0.62225576313444316779, 0.95828213067469025432};

      Y dy = dt * f(t, y);
      y += rk4b[0] * dy;

      for (int stage = 1; stage < 5; ++stage)
      {
        const real s = t + rk4c[stage] * dt;
        dy = rk4a[stage] * dy + dt * f(s, y);
        y += rk4b[stage] * dy;
      }

      t += dt;
    }
  };

  // specialization of LSRK4 for vector types.
  template <std::ranges::forward_range Y>
  class LSRK4<Y>
  {
  public:
    using real = numcepts::precision_t<Y>;

    template <typename... Args>
    LSRK4(Args &&...args) : p(args...), dy(args...) {}

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
      constexpr real rk4a[5] = {0.0, -0.41789047449985196221, -1.1921516946426769261, -1.6977846924715278362, -1.5141834442571557816};
      constexpr real rk4b[5] = {0.14965902199922911733, 0.37921031299962728091, 0.82295502938698171717, 0.69945045594912210704, 0.15305724796815199267};
      constexpr real rk4c[5] = {0.0, 0.14965902199922911733, 0.37040095736420477295, 0.62225576313444316779, 0.95828213067469025432};

      f(p, t, y);

      auto yi = std::begin(y);
      auto pi = std::begin(p);
      for (auto di = std::begin(dy); di != std::end(dy); ++yi, ++pi, ++di)
      {
        (*di) = dt * (*pi);
        (*yi) += rk4b[0] * (*di);
      }

      for (int stage = 1; stage < 5; ++stage)
      {
        const real s = t + rk4c[stage] * dt;
        f(p, s, y);

        yi = std::begin(y);
        pi = std::begin(p);
        for (auto di = std::begin(dy); di != std::end(dy); ++yi, ++pi, ++di)
        {
          (*di) = rk4a[stage] * (*di) + dt * (*pi);
          (*yi) += rk4b[stage] * (*di);
        }
      }

      t += dt;
    }

  private:
    mutable Y p;
    mutable Y dy;
  };

  // specialization of LSRK4 for arrays (pointers) of scalar types.
  template <numcepts::ScalarType Y>
  class LSRK4<Y *>
  {
  public:
    using real = numcepts::precision_t<Y>;

    LSRK4(size_t dim) : n{dim}, p(dim), dy(dim) {}

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
      constexpr real rk4a[5] = {0.0, -0.41789047449985196221, -1.1921516946426769261, -1.6977846924715278362, -1.5141834442571557816};
      constexpr real rk4b[5] = {0.14965902199922911733, 0.37921031299962728091, 0.82295502938698171717, 0.69945045594912210704, 0.15305724796815199267};
      constexpr real rk4c[5] = {0.0, 0.14965902199922911733, 0.37040095736420477295, 0.62225576313444316779, 0.95828213067469025432};

      f(p.data(), t, y);

      for (size_t i = 0; i < n; ++i)
      {
        dy[i] = dt * p[i];
        y[i] += rk4b[0] * dy[i];
      }

      for (int stage = 1; stage < 5; ++stage)
      {
        const real s = t + rk4c[stage] * dt;
        f(p.data(), s, y);

        for (size_t i = 0; i < n; ++i)
        {
          dy[i] = rk4a[stage] * dy[i] + dt * p[i];
          y[i] += rk4b[stage] * dy[i];
        }
      }

      t += dt;
    }

  private:
    const size_t n;
    mutable std::vector<Y> p;
    mutable std::vector<Y> dy;
  };
} // namespace ivp

#endif