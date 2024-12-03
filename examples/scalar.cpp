/**
 * @file scalar.cpp
 * @brief Example driver for solving a scalar initial value problem using a
 * fixed step size.
 *
 * Here we solve:
 *  y' = y / (t + 1) - t^2 * y^2,    y(0) = y0.
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

// y' = F(t, y).
double f(double t, double y)
{
  return y / (t + 1.0) - (t * t) * (y * y);
}

// the exact solution y(t) given the initial condition y(0) = y0.
double exact(double t, double y0)
{
  double p = 12.0 * y0 * (1.0 + t);
  double q = 12.0 + y0 * t * t * t * (4.0 + 3.0 * t);
  return p / q;
}

int main()
{
  double t = 0;
  double y0 = 1;
  double y = y0;

  const double T = 3.0;
  const int nt = 100;

  const double dt = T / nt;

  // uncomment one of the solvers:

  // ivp::ForwardEuler<double> rk;
  // ivp::RK2<double> rk;
  // ivp::SSPRK3<double> rk;
  ivp::LSRK4<double> rk;

  double err = 0;

  std::cout << std::setprecision(10)
            << std::setw(5) << "#"
            << std::setw(20) << "t"
            << std::setw(20) << "y"
            << std::setw(20) << "error" << "\n"
            << std::setw(5) << 0
            << std::fixed << std::setw(20) << t
            << std::setw(20) << y
            << std::scientific << std::setw(20) << y - exact(t, y0)
            << "\n";

  for (int it = 1; it <= nt; ++it)
  {
    rk.step(dt, f, t, y);

    double y_exact = exact(t, y0);
    double e = std::abs(y - exact(t, y0));

    std::cout << std::setw(5) << it
              << std::fixed << std::setw(20) << t
              << std::setw(20) << y
              << std::scientific << std::setw(20) << e
              << "\n";

    err = std::max(err, e);
  }

  std::cout << "Max error = " << std::setprecision(2) << std::scientific << err << "\n";

  return 0;
}