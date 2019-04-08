# libsvmapp
libsvm是基于文件作为scale/train/predict的数据输入输出媒介，因频繁系统调用io导致性能差，本例使用内存buffer作为数据传输媒介，大幅提高predict的性能。
libsvm官网https://www.csie.ntu.edu.tw/~cjlin/libsvm/
