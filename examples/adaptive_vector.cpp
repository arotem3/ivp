/**
 * @file adaptive_vector.cpp
 * @brief Example driver for solving a vector valued initial value problem with
 * adaptive step size control. Here the vector is represented as an std::vector.
 *
 * Here we solve:
 *  x' = -t x - y,    x[0] = x0,
 *  y' = 4 x - t y,   y[0] = y0.
 *
 * We compare the computed solution to the exact solution.
 *
 * @copyright Copyright (c) 2024
 *
 */

#include <cmath>
#include <iostream>
#include <iomanip>

#include "../ivp.hpp"

using dvec = std::vector<double>;

// y' = p = F(t, y)
void f(dvec &p, double t, const dvec &y)
{
  p[0] = -t * y[0] - y[1];
  p[1] = 4.0 * y[0] - t * y[1];
}

// the exact solution y(t) given the initial condition y(0) = y0.
void exact(dvec &y, double t, const dvec &y0)
{
  double c = std::cos(2.0 * t);
  double s = std::sin(2.0 * t);
  double r = std::exp(-0.5 * t * t);

  y[0] = (y0[0] * c - 0.5 * y0[1] * s) * r;
  y[1] = (y0[1] * c + 2.0 * y0[0] * s) * r;
}

int main()
{
  double t = 0;
  dvec y0 = {1.0, -2.0};
  dvec y = {y0[0], y0[1]};
  dvec p(2);
  f(p, t, y);

  const double T = 3.0;
  double dt = 0.05;

  // uncomment one of the solvers:

  // ivp::ARK3E2<dvec> rk({}, 2);
  ivp::ARK5E4<dvec> rk({.rtol = 1e-5, .atol = 1e-8}, 2);

  int it = 0;
  double err = 0;
  dvec y_exact(2);
  exact(y_exact, t, y0);

  std::cout << std::setprecision(10)
            << std::setw(5) << "#"
            << std::setw(20) << "t"
            << std::setw(20) << "y[0]"
            << std::setw(20) << "y[1]"
            << std::setw(20) << "error" << "\n"
            << std::setw(5) << it
            << std::fixed << std::setw(20) << t
            << std::setw(20) << y[0]
            << std::setw(20) << y[1]
            << std::scientific << std::setw(20) << std::hypot(y_exact[0] - y[0], y_exact[1] - y[1])
            << "\n";

  while (t < T)
  {
    it++;
    rk.step(dt, f, t, y, p);
    dt = std::min(dt, T - t);

    exact(y_exact, t, y0);
    double e = std::hypot(y_exact[0] - y[0], y_exact[1] - y[1]);

    std::cout << std::setw(5) << it
              << std::fixed << std::setw(20) << t
              << std::setw(20) << y[0]
              << std::setw(20) << y[1]
              << std::scientific << std::setw(20) << e
              << "\n";

    err = std::max(err, e);
  }

  std::cout << "Max error = " << std::setprecision(2) << std::scientific << err << "\n";

  return 0;
}