cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 17)
PROJECT(test)

add_executable(test main.cpp )
target_link_libraries(test  ${CMAKE_DL_LIBS})
target_compile_definitions(test PRIVATE USE_TINY )
target_include_directories(test PRIVATE .  )

