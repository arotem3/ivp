# `ivp.hpp`
Small selection of generic one step methods for solving initial value problems:
$$y'(t) = F(t,\, y(t)), \quad y(t_0) = y_0.$$
Here $y : \mathbb{R} \to V$. Where $V$ is an $n$ dimensional vector field, i.e. $\mathbb{R}^n$ or $\mathbb{C}^n$.

## Explicit Methods
These methods are typically used when the time step is fixed, or the time step is adapted according using, say, Richardson extrapolation. All of these methods requires `2n` storage (where `n` is the dimension of the system of ODEs).

* `ForwardEuler`: implements the first order [forward Euler method](https://en.wikipedia.org/wiki/Euler_method). Requires one function evaluation every step.
* `RK2`: implments the [explicit midpoint method](https://en.wikipedia.org/wiki/Midpoint_method). Requires two function evaluations every step.
* `SSPRK3`: implements a three stage third order strong stability preserving Runge Kutta method. Requires three function evaluations every step. See [_Strong stability preserving high order time discretization methods._ S. Gottlieb, C.-W. Shu, and E. Tadmor](https://doi.org/10.1137/S003614450036757X)
* `LSRK4`: implements a five stage fourth order Runge Kutta method. Requires five function evaluations every step. This is not the classic fourth order method of Kutta, but rather the low storage method of Carpenter and Kennedy. This method has less storage requirements and a larger stability region than the classic method. See [_Fourth-order 2N-storage Runge-Kutta schemes._ M.H. Carpenter and C. Kennedy](https://ntrs.nasa.gov/api/citations/19940028444/downloads/19940028444.pdf)

## Adaptive Explicit Methods
These methods use an adaptive step size to ensure that the local error is bounded by a specified tolerance. To determine the step size, these methods use an embedded method of lower order (sometimes higher order) to approximate the local error. As implemented here, if a time step fails the error criteria, a new smaller time step will be computed and the error is estimated again. This procedure is repeated until the error is sufficiently small or fails to produce an estimate if the time step becomes too small. For this reason, the cost of each time step is not predictable. However, for non-stiff problems, most time steps will only require one or two estimations. Once a time step is computed, these methods suggest a new time step for the next step.

* `ARK3E2`: implments a four stage third order method with an embedded second order method. This method has the FSAL (first same as last) property, meaning each time step evaluates $F(t, y(t))$ and $F(t + \Delta t, y(t + \Delta t))$ so that function evaluations may be reused between time steps. Three function evaluations are needed at each time step. This method uses `4n` storage. See [_A 3(2) pair of Runge - Kutta formulas_. P. Bogacki, L.F. Shampine.](https://doi.org/10.1016/0893-9659(89)90079-7)
* `ARK5E4`: implements a seven stage fifth order method with an embedded fourth order method. This method has the FSAL property and requires six function evaluations at each time step. This method uses `8n` storage. See [_A Family of Embedded Runge-Kutta Formulae_. J.R. Dormand, P.J. Prince.](https://doi.org/10.1016/0771-050X(80)90013-3)

The relative error tolerance, absolute error tolerance, minimum step size, and maximum step can be specified by passing an `ARKOpts` instance to the constructor of these adaptive Runge Kutta classes.

## C++ types for the variable $y$.

All of the methods specified have specializations for three abstract [(concept)](https://en.cppreference.com/w/cpp/language/constraints) types:

* `numcepts::ScalarType`: these are any types which behave like a mathematical scalar ($\mathbb{R}$ or $\mathbb{C}$). Namely, `float, double, long double` or `std::complex<...>` all satisfy these requirements.
  * These types may be any type for which `numcepts::is_real<T>` or `numcepts::is_complex<T>` is equivalent to an `std::true_type`. Therefore, a custom type `T` may satisfy this constraint if one of these types is overload to `std::true_type`. In principle, such a type should be default initializable and implement `+`, `-`, `*`, `/` operators which behave like these operations on the real or complex numbers.
  * As an example, `std::valarray<double>` will behave as expected with any of the explicit `ivp` methods by adding the declaration: `struct numcepts::is_real<std::valarray<double>> : std::true_type {};` Nevertheless, the implementation for `std::ranges::forward_range` may be more efficient for `std::valarray<double>` for larger `n`.
* `numcepts::ScalarType[]`: this is an array whose elements are `numcepts::ScalarType`. The size of these arrays is passed to the constructor of the IVP solver and the solver allocates the needed work arrays using `std::vector`s.
* `std::ranges::forward_range`: this is any container with `begin` and `end` and over which we can iterate several times. It is implicitly assumed that the elements of these ranges satsify the `numcepts::ScalarType` constraint so that all operations on these ranges is logical. Many `std` containers naturally satisfy this condition (and namely `std::vector`), so this implementation is the most generic for vector valued initial value problems. Since all of the methods requires some amount of extra storage, the IVP solver must be able to store the type and it must be initialized correctly on construction of the IVP solver.
  * For example, if using an `std::vector<double>`, the size of the vector must be specified to the constructor of the IVP solver. The size is then passed to all of the work variables needed (e.g. 2 vectors are needed for `ForwardEuler` and 8 are needed for `ARK5E4`). If instead we are using `std::array<double, 5>` nothing needs to be passed to the constructor.

Additional constraints for $y$-types:

* `numcepts::get_precision` must be overloaded for this type. This specifies the underlying floating point (or perhaps other representation) of the type. This type is already defined for all standard floating point types, all `std::complex` types, and any type satisfying `std::ranges::forward_range` constraint with a member type `::value_type` which is a floating point type or complex type (many `std` containers satisfy these conditions).