# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/lxai/wrwork/cplusdir/ncnntest15

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lxai/wrwork/cplusdir/ncnntest15/build

# Include any dependencies generated for this target.
include CMakeFiles/onte.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/onte.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/onte.dir/flags.make

CMakeFiles/onte.dir/main.cpp.o: CMakeFiles/onte.dir/flags.make
CMakeFiles/onte.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lxai/wrwork/cplusdir/ncnntest15/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/onte.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/onte.dir/main.cpp.o -c /home/lxai/wrwork/cplusdir/ncnntest15/main.cpp

CMakeFiles/onte.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/onte.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lxai/wrwork/cplusdir/ncnntest15/main.cpp > CMakeFiles/onte.dir/main.cpp.i

CMakeFiles/onte.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/onte.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lxai/wrwork/cplusdir/ncnntest15/main.cpp -o CMakeFiles/onte.dir/main.cpp.s

# Object files for target onte
onte_OBJECTS = \
"CMakeFiles/onte.dir/main.cpp.o"

# External object files for target onte
onte_EXTERNAL_OBJECTS =

onte: CMakeFiles/onte.dir/main.cpp.o
onte: CMakeFiles/onte.dir/build.make
onte: /usr/local/lib/libopencv_highgui.so.4.5.4
onte: /usr/local/lib/libopencv_ml.so.4.5.4
onte: /usr/local/lib/libopencv_objdetect.so.4.5.4
onte: /usr/local/lib/libopencv_photo.so.4.5.4
onte: /usr/local/lib/libopencv_stitching.so.4.5.4
onte: /usr/local/lib/libopencv_video.so.4.5.4
onte: /usr/local/lib/libopencv_videoio.so.4.5.4
onte: /home/lxai/tool/ncnntool/ncnn/build/install/lib/libncnn.a
onte: /usr/local/lib/libopencv_imgcodecs.so.4.5.4
onte: /usr/local/lib/libopencv_calib3d.so.4.5.4
onte: /usr/local/lib/libopencv_dnn.so.4.5.4
onte: /usr/local/lib/libopencv_features2d.so.4.5.4
onte: /usr/local/lib/libopencv_flann.so.4.5.4
onte: /usr/local/lib/libopencv_imgproc.so.4.5.4
onte: /usr/local/lib/libopencv_core.so.4.5.4
onte: /usr/lib/gcc/x86_64-linux-gnu/10/libgomp.so
onte: /usr/lib/x86_64-linux-gnu/libpthread.so
onte: CMakeFiles/onte.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/lxai/wrwork/cplusdir/ncnntest15/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable onte"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/onte.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/onte.dir/build: onte

.PHONY : CMakeFiles/onte.dir/build

CMakeFiles/onte.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/onte.dir/cmake_clean.cmake
.PHONY : CMakeFiles/onte.dir/clean

CMakeFiles/onte.dir/depend:
	cd /home/lxai/wrwork/cplusdir/ncnntest15/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lxai/wrwork/cplusdir/ncnntest15 /home/lxai/wrwork/cplusdir/ncnntest15 /home/lxai/wrwork/cplusdir/ncnntest15/build /home/lxai/wrwork/cplusdir/ncnntest15/build /home/lxai/wrwork/cplusdir/ncnntest15/build/CMakeFiles/onte.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/onte.dir/depend

