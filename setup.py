import os


def get_build_platform(environ=os.environ):
    platform = environ.get('DEEP_EP_PLATFORM', 'cuda').strip().lower()
    if platform not in ('cuda', 'ascend'):
        raise ValueError('DEEP_EP_PLATFORM must be cuda or ascend')
    return platform


def get_ascend_testing_mode(environ=os.environ):
    return '1' if environ.get('DEEP_EP_ASCEND_TESTING') == '1' else '0'


def get_ascend_release_signal_only(environ=os.environ):
    value = environ.get('DEEP_EP_ASCEND_RELEASE_SIGNAL_ONLY', '1')
    if value not in ('0', '1'):
        raise ValueError(
            'DEEP_EP_ASCEND_RELEASE_SIGNAL_ONLY must be 0 or 1')
    return value


def get_torch_npu_root():
    torch_npu_module = sys.modules.get('torch_npu')
    if torch_npu_module is not None and getattr(torch_npu_module, '__file__', None):
        return Path(torch_npu_module.__file__).resolve().parent
    try:
        torch_npu_spec = importlib.util.find_spec('torch_npu')
    except (ValueError, ImportError):
        torch_npu_spec = None
    if torch_npu_spec is None or torch_npu_spec.origin is None:
        raise RuntimeError('Ascend builds require torch_npu')
    return Path(torch_npu_spec.origin).resolve().parent


if __name__ == '__main__':
    build_platform = get_build_platform()


import ast
import re
import subprocess
import setuptools
import importlib
import sys

from pathlib import Path
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py

current_dir = os.path.dirname(os.path.realpath(__file__))
persistent_env_names = ('EP_JIT_CACHE_DIR', 'EP_JIT_PRINT_COMPILER_COMMAND', 'EP_NUM_TOPK_IDX_BITS', 'EP_NCCL_ROOT_DIR')

def load_find_pkgs():
    # Load discover module without triggering `deep_ep.__init__`
    spec = importlib.util.spec_from_file_location(
        'find_pkgs', os.path.join(current_dir, 'deep_ep', 'utils', 'find_pkgs.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Wheel specific: NVIDIA pip wheels (nvidia-nvshmem-cu12, nvidia-nccl-cu12)
# only ship the SO name of the host library, e.g. `libnvshmem_host.so.3`,
# without the unversioned `libnvshmem_host.so` symlink. So `-l:libnvshmem_host.so`
# (exact-name link) cannot resolve. Resolve the real file name at build time
# and pass it through to the linker instead.
def _find_versioned_so(base_dir, prefix):
    """Return the real filename of the first ``{prefix}.so*`` under ``base_dir/lib``.

    Prefers an unversioned ``{prefix}.so`` symlink when present so we keep
    behaving identically to the Tarball install. Falls back to the SONAME
    file (``{prefix}.so.X``) shipped by pip wheels.
    """
    lib_dir = Path(base_dir).joinpath('lib')
    unversioned = lib_dir / f'{prefix}.so'
    if unversioned.exists():
        return unversioned.name
    for file in sorted(lib_dir.rglob(f'{prefix}.so.*')):
        return file.name
    raise ModuleNotFoundError(f'{prefix}.so not found under {lib_dir}')


def get_nvshmem_host_lib_name(base_dir):
    return _find_versioned_so(base_dir, 'libnvshmem_host')


def get_nccl_lib_name(base_dir):
    return _find_versioned_so(base_dir, 'libnccl')


def get_package_version():
    with open(Path(current_dir) / 'deep_ep' / '__init__.py', 'r') as f:
        version_match = re.search(r'^__version__\s*=\s*(.*)$', f.read(), re.MULTILINE)
    public_version = ast.literal_eval(version_match.group(1))

    # noinspection PyBroadException
    try:
        status_cmd = ['git', 'status', '--porcelain']
        status_output = subprocess.check_output(status_cmd).decode('ascii').strip()
        if status_output:
            print(f'Warning: Git working directory is not clean. Uncommitted changes:\n{status_output}')
            assert False, 'Git working directory is not clean'

        cmd = ['git', 'rev-parse', '--short', 'HEAD']
        revision = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
    except:
        revision = '+local'
    return f'{public_version}{revision}'


class CustomBuildPy(build_py):
    def run(self):
        # Make clusters' cache setting default into `envs.py`
        self.generate_default_envs()

        # Finally, run the regular build
        build_py.run(self)

    def generate_default_envs(self):
        code = '# Pre-installed environment variables\n'
        code += 'persistent_envs = dict()\n'
        # noinspection PyShadowingNames
        for name in persistent_env_names:
            code += f"persistent_envs['{name}'] = '{os.environ[name]}'\n" if name in os.environ else ''

        # Create temporary build directory
        build_include_dir = os.path.join(self.build_lib, 'deep_ep')
        os.makedirs(build_include_dir, exist_ok=True)
        with open(os.path.join(self.build_lib, 'deep_ep', 'envs.py'), 'w') as f:
            f.write(code)


class CMakeExtension(setuptools.Extension):
    def __init__(self, name, cmake_source_dir):
        super().__init__(name, sources=[])
        self.cmake_source_dir = str(Path(cmake_source_dir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, extension):
        import torch
        try:
            import pybind11
        except ModuleNotFoundError as error:
            raise RuntimeError(
                'Ascend builds require pybind11; install it with '
                '`python -m pip install pybind11`.') from error

        extension_path = Path(self.get_ext_fullpath(extension.name)).resolve()
        torch_npu_root = get_torch_npu_root()
        pybind11_dir = Path(pybind11.get_cmake_dir()).resolve()
        output_directory = extension_path.parent
        build_directory = Path(self.build_temp) / extension.name
        build_directory.mkdir(parents=True, exist_ok=True)
        ascend_testing = get_ascend_testing_mode()
        release_signal_only = get_ascend_release_signal_only()

        configure = [
            'cmake', extension.cmake_source_dir,
            '-DDEEP_EP_PLATFORM=ascend',
            f'-DDEEP_EP_ASCEND_TESTING={ascend_testing}',
            f'-DDEEP_EP_ASCEND_RELEASE_SIGNAL_ONLY={release_signal_only}',
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={output_directory}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            f'-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}',
            f'-DTORCH_NPU_ROOT={torch_npu_root}',
            f'-Dpybind11_DIR={pybind11_dir}',
            '-DCMAKE_BUILD_TYPE=Release',
        ]
        subprocess.check_call(configure, cwd=build_directory)
        subprocess.check_call(
            ['cmake', '--build', '.', '--target', '_C', '--parallel', '2'],
            cwd=build_directory)


def make_cuda_extension(define_macros=None):
    from torch.utils.cpp_extension import CUDAExtension

    find_pkgs = load_find_pkgs()

    # TODO: make NVSHMEM and legacy optional
    nvshmem_root_dir = find_pkgs.find_nvshmem_root()
    nccl_root_dir = find_pkgs.find_nccl_root()

    # `128,2417` is used to suppress warnings of `fmt`
    cxx_flags = ['-O3', '-Wno-deprecated-declarations', '-Wno-unused-variable', '-Wno-sign-compare', '-Wno-reorder', '-Wno-attributes']
    nvcc_flags = ['-O3', '-Xcompiler', '-O3', '--extended-lambda', '--diag-suppress=128,2417']
    sources = ['csrc/python_api.cpp', 'csrc/kernels/legacy/layout.cu', 'csrc/kernels/legacy/intranode.cu']
    include_dirs = [f'{current_dir}/deep_ep/include',
                    f'{current_dir}/third-party/fmt/include',
                    '/usr/local/cuda/include/cccl']
    library_dirs = []
    nvcc_dlink = []
    extra_link_args = ['-lcuda']

    # NVSHMEM flags. Use the real on-disk file name (which may be SONAME-only
    # like ``libnvshmem_host.so.3`` when NVSHMEM came from a pip wheel) so
    # that ``-l:NAME`` can resolve. The static device library always ships
    # under its canonical name, so it stays hard-coded.
    sources.extend(['csrc/kernels/legacy/internode.cu', 'csrc/kernels/legacy/internode_ll.cu', 'csrc/kernels/backend/nvshmem.cu'])
    include_dirs.extend([f'{nvshmem_root_dir}/include'])
    library_dirs.extend([f'{nvshmem_root_dir}/lib'])
    nvcc_dlink.extend(['-dlink', f'-L{nvshmem_root_dir}/lib', '-lnvshmem_device'])
    nvshmem_host_lib = get_nvshmem_host_lib_name(nvshmem_root_dir)
    extra_link_args.extend([f'-l:{nvshmem_host_lib}', '-l:libnvshmem_device.a', f'-Wl,-rpath,{nvshmem_root_dir}/lib'])

    # NCCL flags. Same story as NVSHMEM above — pip wheels ship
    # ``libnccl.so.2`` only, so resolve the real name dynamically.
    sources.extend(['csrc/kernels/backend/nccl.cu'])
    include_dirs.extend([f'{nccl_root_dir}/include'])
    nccl_lib = get_nccl_lib_name(nccl_root_dir)
    extra_link_args.extend([f'-l:{nccl_lib}', f'-Wl,-rpath,{nccl_root_dir}/lib'])

    # CUDA driver sources
    sources.extend(['csrc/kernels/backend/cuda_driver.cu'])

    # TODO: remove these
    if int(os.getenv('DISABLE_SM90_FEATURES', 0)):
        # Prefer A100
        os.environ['TORCH_CUDA_ARCH_LIST'] = os.getenv('TORCH_CUDA_ARCH_LIST', '8.0')

        # Disable some SM90 features: FP8, launch methods, and TMA
        cxx_flags.append('-DDISABLE_SM90_FEATURES')
        nvcc_flags.append('-DDISABLE_SM90_FEATURES')

        # Disable internode and low-latency kernels
        assert False, 'Not implemented'
    else:
        # Prefer H800 series
        os.environ['TORCH_CUDA_ARCH_LIST'] = os.getenv('TORCH_CUDA_ARCH_LIST', '9.0')

        # CUDA 12 flags
        nvcc_flags.extend(['-rdc=true', '--ptxas-options=--register-usage-level=10'])

    # Disable LD/ST tricks, as some CUDA version does not support `.L1::no_allocate`
    if os.environ['TORCH_CUDA_ARCH_LIST'].strip() != '9.0':
        assert int(os.getenv('DISABLE_AGGRESSIVE_PTX_INSTRS', 1)) == 1
        os.environ['DISABLE_AGGRESSIVE_PTX_INSTRS'] = '1'

    # Disable aggressive PTX instructions
    if int(os.getenv('DISABLE_AGGRESSIVE_PTX_INSTRS', '1')):
        cxx_flags.append('-DDISABLE_AGGRESSIVE_PTX_INSTRS')
        nvcc_flags.append('-DDISABLE_AGGRESSIVE_PTX_INSTRS')

    # Legacy environment name
    if 'TOPK_IDX_BITS' in os.environ:
        assert 'EP_NUM_TOPK_IDX_BITS' not in os.environ
        os.environ['EP_NUM_TOPK_IDX_BITS'] = os.environ['TOPK_IDX_BITS']

    # Bits of `topk_idx.dtype`, choices are 32 and 64
    if 'EP_NUM_TOPK_IDX_BITS' in os.environ:
        num_topk_idx_bits = int(os.environ['EP_NUM_TOPK_IDX_BITS'])
        cxx_flags.append(f'-DEP_NUM_TOPK_IDX_BITS={num_topk_idx_bits}')
        nvcc_flags.append(f'-DEP_NUM_TOPK_IDX_BITS={num_topk_idx_bits}')

    # Put them together
    extra_compile_args = {
        'cxx': cxx_flags,
        'nvcc': nvcc_flags,
    }
    if len(nvcc_dlink) > 0:
        extra_compile_args['nvcc_dlink'] = nvcc_dlink

    # Summary
    print('Build summary:')
    print(' > Platform: cuda')
    print(f' > Sources: {sources}')
    print(f' > Includes: {include_dirs}')
    print(f' > Libraries: {library_dirs}')
    print(f' > Compilation flags: {extra_compile_args}')
    print(f' > Link flags: {extra_link_args}')
    print(f' > Arch list: {os.environ["TORCH_CUDA_ARCH_LIST"]}')
    print(f' > NVSHMEM path: {nvshmem_root_dir}')
    print(f' > NCCL path: {nccl_root_dir}')
    # Print persistent env variables
    persistent_envs = []
    for name in persistent_env_names:
        if name in os.environ:
            persistent_envs.append((name, os.environ[name]))
    if len(persistent_envs) > 0:
        print(f' > Persistent envs:')
        for k, v in persistent_envs:
            print(f'   > {k}: {v}')
    print()

    return CUDAExtension(name='deep_ep._C',
                         include_dirs=include_dirs,
                         library_dirs=library_dirs,
                         sources=sources,
                         define_macros=define_macros,
                         extra_compile_args=extra_compile_args,
                         extra_link_args=extra_link_args)


def make_ascend_extension():
    return CMakeExtension('deep_ep._C', current_dir)


def make_extension(platform):
    if platform == 'ascend':
        return make_ascend_extension()
    return make_cuda_extension(
        define_macros=[('DEEP_EP_PLATFORM_CUDA', '1')])


if __name__ == '__main__':
    extension = make_extension(build_platform)
    if build_platform == 'cuda':
        from torch.utils.cpp_extension import BuildExtension
        build_extension = BuildExtension
    else:
        build_extension = CMakeBuild

    if build_platform == 'ascend':
        print('Build summary:')
        print(' > Platform: ascend')
        print()

    setuptools.setup(
        name='deep_ep',
        version=get_package_version(),
        packages=setuptools.find_packages(include=['deep_ep', 'deep_ep.*']),
        package_data={
            'deep_ep': [
                'include/deep_ep/**/*',
            ]
        },
        ext_modules=[extension],
        cmdclass={
            'build_ext': build_extension,
            'build_py': CustomBuildPy
        }
    )
