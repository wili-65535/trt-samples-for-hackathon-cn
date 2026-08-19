# PythonPlugin

+ The same as BasicExample, but we make the workflow totally in Python script.

+ These examples show 5 ways (using cuda-python, cupy, torch, triton, numba packages respectively) to make it.

+ We keep a cuda-python example (add_scalar_cuda_python-V2-deprecated.py) to use deprecated class `IPluginV2DynamicExt`.

+ This example is too simple to show the performance differences among the libraries.

+ Two pieces of boilerplate that every Python plugin needs now live in `tensorrt_cookbook.utils_plugin` instead of being copied into each script:
  + `KernelHelper` / `get_kernel(code, device_id, function_name)` — compile a CUDA source string with **NVRTC** at run time and return a kernel handle (CUBIN for the exact SM when NVRTC supports it, PTX otherwise). Used by the two `add_scalar_cuda_python*.py` examples.
  + `wrap_device_pointer(pointer, shape, dtype, owner)` — view the raw device pointer TensorRT passes into `enqueue()` as a CuPy array **without copying**, via `cupy.cuda.UnownedMemory`. Used by `add_scalar_cupy.py`; the same array hands off zero-copy to PyTorch (`torch.as_tensor`) or CuteDSL (`cute.runtime.from_dlpack`).
  + `check_nvrtc_error(result)` — unwrap the `(status, *values)` tuples that `cuda.bindings.{driver,runtime,nvrtc}` return, raising on a non-zero status.

+ TODO:
  + Remove the redundant memory copy in torch / triton example, which need a solution of wrapping a pointer as a torch.tensor.
  + Get rid of using cupy, so remove the examples with suffix "-using-cupy".
  + Fix numba example, now I get error like below.

```txt
[ERROR] Exception thrown from enqueue() LinkerError: [222] Call to cuLinkAddData results in CUDA_ERROR_UNSUPPORTED_PTX_VERSION
ptxas application ptx input, line 9; fatal   : Unsupported .version 8.4; current version is '8.3'
```

+ Steps to run.

```bash
python3 add_scalar_cuda_python-V2-deprecated.py
python3 add_scalar_cuda_python.py
python3 add_scalar_cupy.py
python3 add_scalar_numba.py
python3 add_scalar_torch.py
python3 add_scalar_triton.py
```
