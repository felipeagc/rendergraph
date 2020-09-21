cmake_minimum_required(VERSION 3.1)
project(rendergraph VERSION 1.0 LANGUAGES C CXX)

add_library(
	rendergraph

	rendergraph/rendergraph.c
	rendergraph/rendergraph_ext.c
	rendergraph/vk_mem_alloc.cpp

	rendergraph/rendergraph.h
	rendergraph/rendergraph_ext.h
	rendergraph/spirv.h
	rendergraph/vk_mem_alloc.h
	rendergraph/volk.h)

if (MSVC)
else()
	target_compile_options(rendergraph PRIVATE -Wall -Wextra -Wno-unused-function)
endif()

target_include_directories(rendergraph PUBLIC rendergraph)