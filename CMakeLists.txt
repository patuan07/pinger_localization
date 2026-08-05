cmake_minimum_required(VERSION 3.8)

project(pinger_localization)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}/
)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  pinger_localization/pinger_simulator.py
  pinger_localization/eval_node.py
  pinger_localization/solvers/batch_sliding.py
  pinger_localization/solvers/batch_full.py
  pinger_localization/solvers/particle_filter.py
  pinger_localization/solvers/iterative_ekf.py
  pinger_localization/solvers/iterative_rls.py
  pinger_localization/solvers/iterative_gd.py

  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
