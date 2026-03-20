cmake_minimum_required(VERSION 3.20)  # 适配 llama.cpp 要求

project(QtLlamaChat VERSION 0.1 LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

# 关键：把 C++ 标准从 17 改成 20
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # 禁用编译器扩展，避免语法歧义

# 查找 Qt
find_package(QT NAMES Qt6 Qt5 COMPONENTS Widgets REQUIRED)
find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Widgets REQUIRED)

# 引入 llama.cpp 子模块
add_subdirectory(thirdparty/llama.cpp)

# 收集 src 目录下所有源文件
file(GLOB_RECURSE PROJECT_SOURCES
    src/*.cpp
    src/*.h
    src/*.ui
)

# 资源文件
set(RESOURCE_FILE assets/res.qrc)

# 生成可执行文件
if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    qt_add_executable(QtLlamaChat
        MANUAL_FINALIZATION
        ${PROJECT_SOURCES}
        ${RESOURCE_FILE}
    )
else()
    if(ANDROID)
        add_library(QtLlamaChat SHARED
            ${PROJECT_SOURCES}
            ${RESOURCE_FILE}
        )
    else()
        add_executable(QtLlamaChat
            ${PROJECT_SOURCES}
            ${RESOURCE_FILE}
        )
    endif()
endif()

# 链接 Qt + llama.cpp
target_link_libraries(QtLlamaChat PRIVATE
    Qt${QT_VERSION_MAJOR}::Widgets
    llama
)

# 包含头文件路径
target_include_directories(QtLlamaChat PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/llama.cpp
)
if(WIN32)
    add_custom_command(TARGET QtLlamaChat POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:llama>
        $<TARGET_FILE:ggml>
        $<TARGET_FILE:ggml-base>
        $<TARGET_FILE:ggml-cpu>
        $<TARGET_FILE_DIR:QtLlamaChat>
    )
endif()

# 平台相关设置
set_target_properties(QtLlamaChat PROPERTIES
    MACOSX_BUNDLE_GUI_IDENTIFIER my.example.com
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(QtLlamaChat)
endif()
