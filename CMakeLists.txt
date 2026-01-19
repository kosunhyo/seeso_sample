cmake_minimum_required(VERSION 3.5)
project(eyedid_cpp_sample)

set(CMAKE_CXX_STANDARD 11)

add_subdirectory(opencv)
add_subdirectory(eyedid)

add_executable(eyedid_cpp_sample main.cpp
        tracker_manager.cc
        camera_thread.cc
        view.cc
        priority_mutex.cc
        kalman_filter.cc)

target_link_libraries(eyedid_cpp_sample PUBLIC opencv seeso)

if (DEFINED EYEDID_TEST_KEY)
    ADD_DEFINITIONS(-DEYEDID_TEST_KEY=${EYEDID_TEST_KEY})
endif()

if(WIN32)
# copy seeso dlls
add_custom_command(TARGET eyedid_cpp_sample POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${SEESO_DLL}
        $<TARGET_FILE_DIR:eyedid_cpp_sample>/)

#copy opencv dlls
add_custom_command(TARGET eyedid_cpp_sample POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${OPENCV_DLL}
        $<TARGET_FILE_DIR:eyedid_cpp_sample>/)
else()
    message(AUTHOR_WARNING "Eyedid : Running on non-Windows is experimental")
endif()
