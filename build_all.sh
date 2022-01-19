cd build
cmake -DTORCH_PATH=~/anaconda3/envs/fvsrn/lib/python3.8/site-packages/torch ..
make -j8 VERBOSE=true
cd ..
