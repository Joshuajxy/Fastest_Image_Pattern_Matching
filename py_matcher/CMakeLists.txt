cmake_minimum_required(VERSION 3.12)
project(template_matcher)

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add compile definitions for Windows
if(WIN32)
    add_definitions(-D_WINDOWS)
endif()


# 设置OpenCV路径
set(OpenCV_DIR "C:/Users/joshua.yang/Downloads/opencv/build")

# Anaconda Python配置
set(Python_ROOT_DIR "C:/Users/joshua.yang/anaconda3")
set(Python_EXECUTABLE "C:/Users/joshua.yang/anaconda3/python.exe")
set(Python_INCLUDE_DIR "C:/Users/joshua.yang/anaconda3/include")
set(Python_LIBRARY "C:/Users/joshua.yang/anaconda3/libs/python311.lib")

# 设置输出目录
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Find required packages
find_package(Python 3.11 EXACT COMPONENTS Interpreter Development NumPy REQUIRED)
find_package(OpenCV REQUIRED)

# pybind11
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 CONFIG REQUIRED)

# 核心库源文件
set(CORE_SOURCES
    src/core/template_matcher.cpp
)

# 添加核心库
add_library(matcher_core STATIC ${CORE_SOURCES})

# 设置包含目录
target_include_directories(matcher_core 
    PUBLIC 
        ${OpenCV_INCLUDE_DIRS}
        ${Python_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core
)

# 链接OpenCV
target_link_libraries(matcher_core
    PUBLIC
        ${OpenCV_LIBS}
)

# Python模块
pybind11_add_module(cvmatcher 
    src/python/matcher_wrapper.cpp
)

# 链接核心库到Python模块
target_link_libraries(cvmatcher 
    PRIVATE 
        matcher_core
)

# 安装规则
install(TARGETS matcher_core
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(TARGETS cvmatcher
    LIBRARY DESTINATION ${Python_SITEARCH}
)

# 复制Python示例
install(FILES 
    examples/match_example.py
    DESTINATION ${CMAKE_INSTALL_PREFIX}/examples
)

# 设置VS工程属性
if(MSVC)
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT cvmatcher)
    
    # 设置多处理器编译
    target_compile_options(matcher_core PRIVATE /MP)
    target_compile_options(cvmatcher PRIVATE /MP)
    
    # Debug信息格式
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Zi")
    set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} /DEBUG:FULL")
endif()

# 添加测试支持（可选）
option(BUILD_TESTING "Build the testing tree." OFF)
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()