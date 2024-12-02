#ifndef IVP_ARK3E2_HPP
#define IVP_ARK3E2_HPP

#include <vector>
#include <ranges>
#include "../numcepts/numcepts.hpp"
#include "ARKOpts.hpp"

namespace ivp
{
  /**
   * @brief Implements the adaptive explicit Runge Kutta third order method with
   * an embedded second order method of Bogacki and Shampine.
   *
   * @tparam Y scalar or vector type for y-variable. float, complex<...>,
   * vector<...>, array<...>, etc.
   */
  template <typename Y>
  class ARK3E2;

  // specialization of ARK3E2 for scalar types.
  template <numcepts::ScalarType Y>
  class ARK3E2<Y>
  {
  public:
    using real = numcepts::precision_t<Y>;

    inline explicit ARK3E2(ARKOpts<real> opts = {}) : opts{opts} {}

    /**
     * @brief computes a single step fo the IVP y' = F(t, y)
     *
     * @tparam Func invocable as f(real, const Y &)
     * @param dt Step size. This step size may be rejected and a smaller step
     * will be taken instead. On exit, dt is the suggested step size for the
     * next step.
     * @param f invocable as f(t, y). Returns F(t, y).
     * @param t current t-value. On exit, t = t + dt.
     * @param y current y-value. On exit, y ~ y(t + dt).
     * @param p = F(t, y) for current t and y-values. On exit, p = F(t, y) where
     * t and y are the exit values of t and y.
     *
     * @return true if step was successfully computed, and false if a step could
     * not be computed without taking step smaller than the minimum admissible
     * size.
     */
    template <typename Func>
    bool step(real &dt, Func &&f, real &t, Y &y, Y &p) const
    {
      constexpr real min_factor = 0.2, max_factor = 2.0, safety = 0.9;

      constexpr real c[] = {0.0, 0.5, 0.75, 1.0};
      constexpr real b[] = {2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0, 0.0};
      constexpr real bh[] = {-5.0 / 72.0, 1.0 / 12.0, 1.0 / 9.0, -1.0 / 8.0};

      const Y p0 = p;
      Y u;
      real s;

      bool reject = true;
      while (reject)
      {
        // compute step
        p = p0;
        Y dy = b[0] * dt * p;
        Y e = bh[0] * dt * p;

        for (int stage : {1, 2})
        {
          u = y + c[stage] * dt * p;
          s = t + c[stage] * dt;

          p = f(s, u);

          dy += b[stage] * dt * p;
          e += bh[stage] * dt * p;
        }

        u = y + dy;
        s = t + dt;

        p = f(s, u);

        e += bh[3] * dt * p;

        // determine if error is admissible and update step size
        const real R = opts.atol + std::max(std::abs(y), std::abs(y + dy)) * opts.rtol;
        const real err = std::abs(e / R);

        reject = err > 1;

        const real scale = std::pow(err, real(-1.0 / 3.0));
        dt *= std::min(max_factor, std::max(min_factor, safety * scale));

        dt = std::min(dt, opts.max_dt);
        if (dt < opts.min_dt)
          return false;
      }

      t = s;
      y = u;
      return true;
    }

  private:
    const ARKOpts<real> opts;
  };

  // specialization of ARK3E2 for arrays (pointers) of scalar types.
  template <numcepts::ScalarType Y>
  class ARK3E2<Y *>
  {
  public:
    using real = numcepts::precision_t<Y>;

    inline explicit ARK3E2(size_t dim, ARKOpts<real> opts = {})
        : n{dim}, opts{opts}, p(dim), u(dim), dy(dim), e(dim) {}

    /**
     * @brief computes a single step fo the IVP y' = F(t, y)
     *
     * @tparam Func invocable as f(real, const Y *)
     * @param dt Step size. This step size may be rejected and a smaller step
     * will be taken instead. On exit, dt is the suggested step size for the
     * next step.
     * @param f invocable as f(t, y). Returns F(t, y).
     * @param t current t-value. On exit, t = t + dt.
     * @param y current y-value. On exit, y ~ y(t + dt).
     * @param p = F(t, y) for current t and y-values. On exit, p = F(t, y) where
     * t and y are the exit values of t and y.
     *
     * @return true if step was successfully computed, and false if a step could
     * not be computed without taking step smaller than the minimum admissible
     * size.
     */
    template <typename Func>
    bool step(real &dt, Func &&f, real &t, Y *y, Y *p0) const
    {
      constexpr real min_factor = 0.2, max_factor = 2.0, safety = 0.9;

      constexpr real c[] = {0.0, 0.5, 0.75, 1.0};
      constexpr real b[] = {2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0, 0.0};
      constexpr real bh[] = {-5.0 / 72.0, 1.0 / 12.0, 1.0 / 9.0, -1.0 / 8.0};

      real s;

      bool reject = true;
      while (reject)
      {
        for (size_t i = 0; i < n; ++i)
        {
          p[i] = p0[i];
          dy[i] = (b[0] * dt) * p[i];
          e[i] = (bh[0] * dt) * p[i];
        }

        for (int stage : {1, 2})
        {
          s = t + c[stage] * dt;

          for (size_t i = 0; i < n; ++i)
            u[i] = y[i] + (c[stage] * dt) * p[i];

          f(p.data(), s, u.data());

          for (size_t i = 0; i < n; ++i)
          {
            dy[i] += (b[stage] * dt) * p[i];
            e[i] += (bh[stage] * dt) * p[i];
          }
        }

        s = t + dt;
        for (size_t i = 0; i < n; ++i)
          u[i] = y[i] + dy[i];

        f(p.data(), s, u.data());

        real err = 0;
        for (size_t i = 0; i < n; ++i)
        {
          e[i] += (bh[3] * dt) * p[i];

          const real R = opts.atol + std::max(std::abs(y[i]), std::abs(u[i])) * opts.rtol;
          err += std::pow(std::abs(e[i]) / R, 2);
        }
        err = std::sqrt(err / n);

        reject = err > 1;

        const real scale = std::pow(err, real(-1.0 / 3.0));
        dt *= std::min(max_factor, std::max(min_factor, safety * scale));

        dt = std::min(dt, opts.max_dt);
        if (dt < opts.min_dt)
          return false;
      }

      t = s;
      for (size_t i = 0; i < n; ++i)
      {
        y[i] = u[i];
        p0[i] = p[i];
      }
      return true;
    }

  private:
    const size_t n;

    const ARKOpts<real> opts;

    mutable std::vector<Y> p;
    mutable std::vector<Y> u;
    mutable std::vector<Y> dy;
    mutable std::vector<Y> e;
  };

  // specialization of ARK3E2 for arrays (pointers) of scalar types.
  template <std::ranges::forward_range Y>
  class ARK3E2<Y>
  {
  public:
    using real = numcepts::precision_t<Y>;

    template <typename... Args>
    inline explicit ARK3E2(ARKOpts<real> opts = {}, Args&&... args)
      : opts{opts}, p(args...), u(args...), dy(args...), e(args...)
    {
      n = std::distance(std::begin(p), std::end(p));
    }

    /**
     * @brief computes a single step fo the IVP y' = F(t, y)
     *
     * @tparam Func invocable as f(Y &, real, const Y &)
     * @param dt Step size. This step size may be rejected and a smaller step
     * will be taken instead. On exit, dt is the suggested step size for the
     * next step.
     * @param f invocable as f(p, t, y). On exit, p = F(t, y).
     * @param t current t-value. On exit, t = t + dt.
     * @param y current y-value. On exit, y ~ y(t + dt).
     * @param p = F(t, y) for current t and y-values. On exit, p = F(t, y) where
     * t and y are the exit values of t and y.
     *
     * @return true if step was successfully computed, and false if a step could
     * not be computed without taking step smaller than the minimum admissible
     * size.
     */
    template <typename Func>
    bool step(real &dt, Func &&f, real &t, Y &y, Y &p0) const
    {
      constexpr real min_factor = 0.2, max_factor = 2.0, safety = 0.9;

      constexpr real c[] = {0.0, 0.5, 0.75, 1.0};
      constexpr real b[] = {2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0, 0.0};
      constexpr real bh[] = {-5.0 / 72.0, 1.0 / 12.0, 1.0 / 9.0, -1.0 / 8.0};

      real s;

      bool reject = true;
      while (reject)
      {
        auto p_iter = std::begin(p);
        auto dy_iter = std::begin(dy);
        auto e_iter = std::begin(e);
        for (auto p_value : p0)
        {
          *p_iter = p_value;
          *dy_iter = (b[0] * dt) * p_value;
          *e_iter = (bh[0] * dt) * p_value;

          ++p_iter;
          ++dy_iter;
          ++e_iter;
        }

        for (int stage : {1, 2})
        {
          s = t + c[stage] * dt;

          p_iter = std::begin(p);
          auto u_iter = std::begin(u);
          for (auto y_value : y)
          {
            *u_iter = y_value + (c[stage] * dt) * (*p_iter);

            ++u_iter;
            ++p_iter;
          }

          f(p, s, u);

          dy_iter = std::begin(dy);
          e_iter = std::begin(e);
          for (auto p_value : p)
          {
            *dy_iter += (b[stage] * dt)  * p_value;
            *e_iter  += (bh[stage] * dt) * p_value;

            ++dy_iter;
            ++e_iter;
          }
        }

        s = t + dt;
        auto u_iter = std::begin(u);
        dy_iter = std::begin(dy);
        for (auto y_value : y)
        {
          *u_iter = y_value + (*dy_iter);

          ++u_iter;
          ++dy_iter;
        }

        f(p, s, u);

        real err = 0;
        auto y_iter = std::begin(y);
        u_iter = std::begin(u);
        p_iter = std::begin(p);
        for (auto &e_value : e)
        {
          e_value += (bh[3] * dt) * (*p_iter);

          const real R = opts.atol + std::max(std::abs(*y_iter), std::abs(*u_iter)) * opts.rtol;
          err += std::pow(std::abs(e_value) / R, 2);

          ++y_iter;
          ++u_iter;
          ++p_iter;
        }
        err = std::sqrt(err / n);

        reject = err > 1;

        const real scale = std::pow(err, real(-1.0 / 3.0));
        dt *= std::min(max_factor, std::max(min_factor, safety * scale));

        dt = std::min(dt, opts.max_dt);
        if (dt < opts.min_dt)
          return false;
      }

      t = s;

      auto y_iter = std::begin(y);
      auto p_iter = std::begin(p0);
      auto u_iter = std::begin(u);
      for (auto p_value : p)
      {
        *y_iter = *u_iter;
        *p_iter = p_value;

        ++y_iter;
        ++p_iter;
        ++u_iter;
      }

      return true;
    }

  private:
    size_t n;

    const ARKOpts<real> opts;

    mutable Y p;
    mutable Y u;
    mutable Y dy;
    mutable Y e;
  };
  
} // namespace ivp

#endif
