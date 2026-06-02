cmake_minimum_required(VERSION 3.16)
project(tinyinfer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ---------- core library ----------
add_library(tinyinfer
    src/matmul.cpp
    src/softmax.cpp
    src/layernorm.cpp
)
target_include_directories(tinyinfer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ---------- tests ----------
option(TINYINFER_BUILD_TESTS "Build tests" ON)
if(TINYINFER_BUILD_TESTS)
    enable_testing()

    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    file(GLOB TEST_SOURCES "tests/test_*.cpp")
    foreach(test_src ${TEST_SOURCES})
        get_filename_component(test_name ${test_src} NAME_WE)
        add_executable(${test_name} ${test_src})
        target_link_libraries(${test_name} PRIVATE tinyinfer GTest::gtest_main)
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()
endif()
