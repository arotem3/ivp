##### --- examples --- #####

##### adaptive_array #####
add_executable(adaptive_array adaptive_array.cpp)
target_link_libraries(adaptive_array PUBLIC ivp)

##### adaptive_scalar #####
add_executable(adaptive_scalar adaptive_scalar.cpp)
target_link_libraries(adaptive_scalar PUBLIC ivp)

##### adaptive_vector #####
add_executable(adaptive_vector adaptive_vector.cpp)
target_link_libraries(adaptive_vector PUBLIC ivp)

##### array #####
add_executable(array array.cpp)
target_link_libraries(array PUBLIC ivp)

##### scalar #####
add_executable(scalar scalar.cpp)
target_link_libraries(scalar PUBLIC ivp)

##### vector #####
add_executable(vector vector.cpp)
target_link_libraries(vector PUBLIC ivp)
