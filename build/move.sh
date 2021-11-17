source scl_source enable devtoolset-10
CC=gcc CXX=g++ cmake ..
#cmake .. -DCMAKE_BUILD_TYPE=Release #Release Debug
cmake --build . -j12

# rm /pvfsmnt/119010114/ass3/csc4005_imgui
# rm /pvfsmnt/119010114/ass3/testMPI
# rm /pvfsmnt/119010114/ass3/testP
rm /pvfsmnt/119010114/ass3/testCUDA
rm /pvfsmnt/119010114/ass3/test.sh
# rm /pvfsmnt/119010114/ass3/testOMP
cp /home/119010114/csc4005-assignment-3_cuda/csc4005-imgui/build/test.sh /home/119010114/csc4005-assignment-3_cuda/csc4005-imgui/build/testCUDA -d /pvfsmnt/119010114/ass3


# then call mpirun-gui for mpi test
