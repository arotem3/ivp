#ifndef IVP_ARK5E4_HPP
#define IVP_ARK5E4_HPP

#include <vector>
#include <ranges>
#include "../numcepts/numcepts.hpp"
#include "ARKOpts.hpp"

namespace ivp
{
  /**
   * @brief Implements the adaptive explicit Runge Kutta seven stage fifth order
   * method with a fourth order embedded method of Dormand and Prince.
   *
   * @tparam Y scalar or vector type for y-variable. float, complex<...>,
   * vector<...>, array<...>, etc.
   */
  template <typename Y>
  class ARK5E4;

  // specialization of ARK5E4 for scalar types.
  template <numcepts::ScalarType Y>
  class ARK5E4<Y>
  {
  public:
    using real = numcepts::precision_t<Y>;

    inline explicit ARK5E4(ARKOpts<real> opts = {}) : opts{opts} {}

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
      constexpr real a[6][6] = {
          {0.20000000000000000000, 0, 0, 0, 0, 0},
          {0.07500000000000000000, 0.2250000000000000000, 0, 0, 0, 0},
          {0.97777777777777777778, -3.7333333333333333333, 3.5555555555555555556, 0, 0, 0},
          {2.9525986892242036275, -11.595793324188385917, 9.8228928516994360616, -0.29080932784636488340, 0, 0},
          {2.8462752525252525253, -10.757575757575757576, 8.9064227177434724605, 0.27840909090909090909, -0.27353130360205831904, 0},
          {0.091145833333333333333, 0, 0.44923629829290206649, 0.65104166666666666667, -0.32237617924528301887, 0.13095238095238095238}};
      constexpr real c[6] = {0.2, 0.3, 0.8, 8.0 / 9.0, 1, 1};
      constexpr real b[7] = {0.0012326388888888888889, 0, -0.0042527702905061395627, 0.036979166666666666667, -0.050863797169811320755, 0.041904761904761904762, -0.025000000000000000000};

      Y K[6];

      Y u;
      real s;

      bool reject = true;
      while (reject)
      {
        Y e = dt * b[0] * p;
        for (int stage = 0; stage < 6; ++stage)
        {
          s = t + c[stage] * dt;
          u = y + a[stage][0] * dt * p;
          for (int j = 1; j <= stage; ++j)
            u += a[stage][j] * dt * K[j - 1];
          K[stage] = f(s, u);
          e += dt * b[stage + 1] * K[stage];
        }

        const real R = opts.atol + std::max(std::abs(y), std::abs(u)) * opts.rtol;
        const real err = std::abs(e / R);

        reject = err > 1;
        const real scale = std::pow(err, real(-0.2));
        dt *= std::min(max_factor, std::max(min_factor, safety * scale));

        dt = std::min(dt, opts.max_dt);
        if (dt < opts.min_dt)
          return false;
      }

      t = s;
      y = u;
      p = K[5];
      return true;
    }

  private:
    ARKOpts<real> opts;
  };

  // specialization of ARK5E4 for arrays (pointers) of scalar types.
  template <numcepts::ScalarType Y>
  class ARK5E4<Y *>
  {
  public:
    using real = numcepts::precision_t<Y>;

    inline explicit ARK5E4(size_t dim, ARKOpts<real> opts = {}) : n{dim}, opts{opts}, u(dim), e(dim)
    {
      for (int i = 0; i < 6; ++i)
        K[i].resize(dim);
    }

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
    bool step(real &dt, Func &&f, real &t, Y *y, Y *p) const
    {
      constexpr real min_factor = 0.2, max_factor = 2.0, safety = 0.9;
      constexpr real a[6][6] = {
          {0.20000000000000000000, 0, 0, 0, 0, 0},
          {0.07500000000000000000, 0.2250000000000000000, 0, 0, 0, 0},
          {0.97777777777777777778, -3.7333333333333333333, 3.5555555555555555556, 0, 0, 0},
          {2.9525986892242036275, -11.595793324188385917, 9.8228928516994360616, -0.29080932784636488340, 0, 0},
          {2.8462752525252525253, -10.757575757575757576, 8.9064227177434724605, 0.27840909090909090909, -0.27353130360205831904, 0},
          {0.091145833333333333333, 0, 0.44923629829290206649, 0.65104166666666666667, -0.32237617924528301887, 0.13095238095238095238}};
      constexpr real c[6] = {0.2, 0.3, 0.8, 8.0 / 9.0, 1, 1};
      constexpr real b[7] = {0.0012326388888888888889, 0, -0.0042527702905061395627, 0.036979166666666666667, -0.050863797169811320755, 0.041904761904761904762, -0.025000000000000000000};

      real s;

      bool reject = true;
      while (reject)
      {
        for (size_t i = 0; i < n; ++i)
          e[i] = (dt * b[0]) * p[i];

        for (int stage = 0; stage < 6; ++stage)
        {
          s = t + c[stage] * dt;
          for (size_t i = 0; i < n; ++i)
          {
            u[i] = y[i] + (a[stage][0] * dt) * p[i];
            for (int j = 1; j <= stage; ++j)
              u[i] += (a[stage][j] * dt) * K[j - 1][i];
          }

          f(K[stage].data(), s, u.data());

          for (size_t i = 0; i < n; ++i)
            e[i] += (dt * b[stage + 1]) * K[stage][i];
        }

        real err = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
          const real R = opts.atol + std::max(std::abs(y[i]), std::abs(u[i])) * opts.rtol;
          err += std::pow(std::abs(e[i]) / R, 2);
        }
        err = std::sqrt(err / n);

        reject = err > 1;
        const real scale = std::pow(err, real(-0.2));
        dt *= std::min(max_factor, std::max(min_factor, safety * scale));

        dt = std::min(dt, opts.max_dt);
        if (dt < opts.min_dt)
          return false;
      }

      t = s;
      for (size_t i = 0; i < n; ++i)
      {
        y[i] = u[i];
        p[i] = K[5][i];
      }

      return true;
    }

  private:
    const size_t n;

    const ARKOpts<real> opts;

    mutable std::vector<Y> u;
    mutable std::vector<Y> e;
    mutable std::vector<Y> K[6];
  };

  template <std::ranges::forward_range Y>
  class ARK5E4<Y>
  {
  public:
    using real = numcepts::precision_t<Y>;

    template <typename... Args>
    inline explicit ARK5E4(ARKOpts<real> opts = {}, Args &&...args)
        : opts{opts}, u(args...), e(args...), K{Y(args...), Y(args...), Y(args...), Y(args...), Y(args...), Y(args...)}
    {
      n = std::distance(std::begin(u), std::end(u));
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
    bool step(real &dt, Func &&f, real &t, Y &y, Y &p) const
    {
      constexpr real min_factor = 0.2, max_factor = 2.0, safety = 0.9;
      constexpr real a[6][6] = {
          {0.20000000000000000000, 0, 0, 0, 0, 0},
          {0.07500000000000000000, 0.2250000000000000000, 0, 0, 0, 0},
          {0.97777777777777777778, -3.7333333333333333333, 3.5555555555555555556, 0, 0, 0},
          {2.9525986892242036275, -11.595793324188385917, 9.8228928516994360616, -0.29080932784636488340, 0, 0},
          {2.8462752525252525253, -10.757575757575757576, 8.9064227177434724605, 0.27840909090909090909, -0.27353130360205831904, 0},
          {0.091145833333333333333, 0, 0.44923629829290206649, 0.65104166666666666667, -0.32237617924528301887, 0.13095238095238095238}};
      constexpr real c[6] = {0.2, 0.3, 0.8, 8.0 / 9.0, 1, 1};
      constexpr real b[7] = {0.0012326388888888888889, 0, -0.0042527702905061395627, 0.036979166666666666667, -0.050863797169811320755, 0.041904761904761904762, -0.025000000000000000000};

      real s;

      bool reject = true;

      while (reject)
      {
        auto e_iter = std::begin(e);
        for (auto p_value : p)
        {
          *e_iter = (dt * b[0]) * p_value;

          ++e_iter;
        }

        for (int stage = 0; stage < 6; ++stage)
        {
          s = t + c[stage] * dt;

          auto u_iter = std::begin(u);
          auto p_iter = std::begin(p);
          typename Y::iterator k_iter[] = {
              std::begin(K[0]), std::begin(K[1]), std::begin(K[2]),
              std::begin(K[3]), std::begin(K[4]), std::begin(K[5])};
          for (auto y_value : y)
          {
            *u_iter = y_value + (a[stage][0] * dt) * (*p_iter);
            for (int j = 1; j <= stage; ++j)
            {
              *u_iter += (a[stage][j] * dt) * (*(k_iter[j - 1]));
              k_iter[j - 1]++;
            }

            ++u_iter;
            ++p_iter;
          }

          f(K[stage], s, u);

          e_iter = std::begin(e);
          for (auto k_value : K[stage])
          {
            *e_iter += (dt * b[stage + 1]) * k_value;

            ++e_iter;
          }
        }

        real err = 0.0;
        auto u_iter = std::begin(u);
        auto y_iter = std::begin(y);
        for (auto e_value : e)
        {
          const real R = opts.atol + std::max(std::abs(*y_iter), std::abs(*u_iter)) * opts.rtol;
          err += std::pow(std::abs(e_value) / R, 2);

          ++u_iter;
          ++y_iter;
        }

        reject = err > 1;
        const real scale = std::pow(err, real(-0.2));
        dt *= std::min(max_factor, std::max(min_factor, safety * scale));

        dt = std::min(dt, opts.max_dt);
        if (dt < opts.min_dt)
          return false;
      }

      t = s;
      auto y_iter = std::begin(y);
      auto u_iter = std::begin(u);
      auto p_iter = std::begin(p);
      for (auto k_value : K[5])
      {
        *y_iter = *u_iter;
        *p_iter = k_value;

        ++y_iter;
        ++u_iter;
        ++p_iter;
      }

      return true;
    }

  private:
    size_t n;

    const ARKOpts<real> opts;

    mutable Y u;
    mutable Y e;
    mutable Y K[6];
  };

} // namespace ivp

#endif