#ifndef IVP_ARK_OPTS_HPP
#define IVP_ARK_OPTS_HPP

#include "../numcepts/numcepts.hpp"
#include <limits>

namespace ivp
{
  /**
   * @brief options for adaptive Runge Kutta solvers.
   * 
   * @tparam real float or double.
   */
  template <numcepts::RealType real>
  struct ARKOpts
  {
    real rtol = 1e-3; /** relative error tolerance */
    real atol = 1e-6; /** absolute error tolerance */
    real min_dt = 0;  /** minimum acceptible step size */
    real max_dt = std::numeric_limits<real>::infinity(); /** maximum acceptible step size */
  };
  
} // namespace ivp


#endif