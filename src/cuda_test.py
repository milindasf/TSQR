import cupy as cp
print("Number of gpu devices : ", cp.cuda.runtime.getDeviceCount())
