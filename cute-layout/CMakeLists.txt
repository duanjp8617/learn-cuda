cmake_minimum_required(VERSION 3.18)

set(TEMPLATE_PROJECT_NAME cute_layout)

# Detect the build machine's first GPU architecture before CUDA compiler setup.
find_program(NVIDIA_SMI_EXECUTABLE nvidia-smi)
if(NVIDIA_SMI_EXECUTABLE)
  execute_process(
    COMMAND "${NVIDIA_SMI_EXECUTABLE}" "--query-gpu=compute_cap" "--format=csv,noheader"
    OUTPUT_VARIABLE CUDA_COMPUTE_CAPABILITY_OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE NVIDIA_SMI_RESULT
  )
  string(REGEX MATCH "^[0-9]+\\.[0-9]+" CUDA_COMPUTE_CAPABILITY "${CUDA_COMPUTE_CAPABILITY_OUTPUT}")
  if(NVIDIA_SMI_RESULT EQUAL 0 AND CUDA_COMPUTE_CAPABILITY)
    string(REPLACE "." "" CMAKE_CUDA_ARCHITECTURES "${CUDA_COMPUTE_CAPABILITY}")
    message(STATUS "Detected CUDA architecture: ${CMAKE_CUDA_ARCHITECTURES}")
  else()
    message(WARNING "Could not detect a CUDA architecture; using CMake's compiler default")
  endif()
else()
  message(WARNING "nvidia-smi was not found; using CMake's compiler default CUDA architecture")
endif()

project(${TEMPLATE_PROJECT_NAME} LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../cutlass" CACHE PATH
    "Path to the CUTLASS source tree")
if(NOT EXISTS "${CUTLASS_DIR}/include/cute/layout.hpp")
  message(FATAL_ERROR "CUTLASS was not found at ${CUTLASS_DIR}")
endif()

set(INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
file(GLOB SRC_FILES CONFIGURE_DEPENDS
  "${SRC_DIR}/*.c"
  "${SRC_DIR}/*.cpp"
  "${SRC_DIR}/*.cc"
  "${SRC_DIR}/*.cxx"
  "${SRC_DIR}/*.cu"
)

add_executable(${PROJECT_NAME})
target_sources(${PROJECT_NAME} PRIVATE ${SRC_FILES})
target_include_directories(${PROJECT_NAME} PRIVATE
  "${INCLUDE_DIR}"
  "${CUTLASS_DIR}/include"
)
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
