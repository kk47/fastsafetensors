# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import platform
import tempfile
import warnings
from typing import Dict, Optional, Tuple

from .. import cpp as _cpp
from .. import fgds_ext as fgds_cpp
from ..common import SafeTensorsMetadata, init_logger, is_gpu_found
from ..frameworks import FrameworkOpBase, TensorBase
from ..st_types import Device, DeviceType, DType
from .base import CopierInterface
from .nogds import load_library_func, new_nogds_file_copier
from .registry import CopierConstructFunc, register_copier_constructor

# Verify the extension actually provides FGDS symbols (Windows stub
# only exposes fgds_is_initialized and none of the classes below).
_FGDS_AVAILABLE = hasattr(fgds_cpp, "fgds_file_handle")

logger = init_logger(__name__)


def _get_device_id(device: Device) -> int:
    device_list = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    idx = device.index if device.index is not None else 0
    return int(device_list.split(",")[idx])


class FgdsFileCopier(CopierInterface):
    def __init__(
        self,
        metadata: SafeTensorsMetadata,
        device: Device,
        reader: fgds_cpp.fgds_file_reader,
        framework: FrameworkOpBase,
        max_threads: int = 16,
    ):
        self.framework = framework
        self.metadata = metadata
        self.device = device
        self.device_id = _get_device_id(device)
        self.copy_reqs: Dict[int, int] = {}
        self.aligned_length = 0
        self.aligned_offset = 0
        self.max_threads = max_threads
        self.o_direct = True

        self.fgds_handle: Optional[fgds_cpp.fgds_file_handle] = None
        self.fgds_reader = reader
        self.registered_memory: Optional[Tuple[int, int, int]] = None

        result = fgds_cpp.fgds_open(self.device_id)
        if result != 0:
            warnings.warn(f"fgds_open failed with error code: {result}", UserWarning)

    def submit_io(
        self, use_buf_register: bool, max_copy_block_size: int
    ) -> _cpp.gds_device_buffer:
        ALIGN = 64 * 1024
        offset = self.metadata.header_length
        length = self.metadata.size_bytes - self.metadata.header_length
        head_bytes = offset % ALIGN
        tail_bytes = (length + head_bytes) % ALIGN
        if tail_bytes > 0:
            tail_bytes = ALIGN - tail_bytes
            aligned_length = length + head_bytes + tail_bytes
        else:
            aligned_length = length + head_bytes
        aligned_offset = offset - head_bytes

        gbuf = self.framework.alloc_tensor_memory(aligned_length, self.device)
        gbuf_ptr = gbuf.get_base_address()

        self.fgds_handle = fgds_cpp.fgds_file_handle(
            self.metadata.src, self.o_direct, self.device_id
        )

        result = fgds_cpp.fgds_regmem(self.device_id, gbuf_ptr, aligned_length, None)
        if result != 0:
            warnings.warn(f"fgds_regmem failed with error code: {result}", UserWarning)
            self.registered_memory = None
        else:
            self.registered_memory = (self.device_id, gbuf_ptr, aligned_length)

        self.copy_reqs = {}
        count = 0
        while count < aligned_length:
            req_len = aligned_length - count
            if req_len > max_copy_block_size:
                req_len = max_copy_block_size
            dev_buf = fgds_cpp.fgds_device_buffer(gbuf_ptr + count, req_len)
            req_id = self.fgds_reader.submit_read(
                self.fgds_handle, dev_buf, aligned_offset + count, req_len, count
            )
            self.copy_reqs[req_id] = count
            count += req_len

        self.aligned_offset = aligned_offset
        self.aligned_length = aligned_length
        self.gbuf = gbuf
        return gbuf

    def wait_io(
        self,
        gbuf: _cpp.gds_device_buffer,
        dtype: DType = DType.AUTO,
        noalign: bool = False,
    ) -> Dict[str, TensorBase]:
        for req_id in sorted(self.copy_reqs.keys()):
            result = self.fgds_reader.wait_read(req_id)
            if result < 0:
                warnings.warn(f"fgds_read failed for request {req_id}", UserWarning)

        if self.registered_memory:
            device_id, addr, size = self.registered_memory
            result = fgds_cpp.fgds_deregmem(device_id, addr, size)
            if result != 0:
                warnings.warn(
                    f"fgds_deregmem failed with error code: {result}", UserWarning
                )
            self.registered_memory = None

        self.copy_reqs = {}

        return self.metadata.get_tensors(
            gbuf, self.device, self.aligned_offset, dtype=dtype
        )


_inited_fgds = False


def init_fgds(framework: Optional[FrameworkOpBase] = None):
    load_library_func(framework)
    global _inited_fgds
    if not _inited_fgds:
        _inited_fgds = True


@register_copier_constructor("fgds", FgdsFileCopier)
def new_fgds_file_copier(
    device: Device,
    framework: FrameworkOpBase,
    bbuf_size_kb: int = 16 * 1024,
    max_threads: int = 16,
    **kwargs,
) -> CopierConstructFunc:
    # FGDS (libfgds.so) is Linux-only; bail out early on other platforms.
    if not _FGDS_AVAILABLE or platform.system() != "Linux":
        warnings.warn(
            "FGDS is not available on this platform. Falling back to NoGDS.",
            UserWarning,
        )
        return new_nogds_file_copier(
            device, bbuf_size_kb, max_threads, framework=framework, **kwargs
        )

    # Load the GPU runtime library first (like GDS does) so that
    # is_gpu_found() can detect CUDA/HIP devices correctly.
    init_fgds(framework)

    device_is_not_cpu = device.type != DeviceType.CPU
    if device_is_not_cpu and not is_gpu_found():
        warnings.warn(
            "GPU runtime library (libcudart.so or libamdhip64.so) not found. "
            "FGDS requires GPU runtime; falling back to NoGDS.",
            UserWarning,
        )
        return new_nogds_file_copier(
            device, bbuf_size_kb, max_threads, framework=framework, **kwargs
        )

    device_id = _get_device_id(device)

    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(b"test")
            tmp_filename = tmp_file.name

        try:
            tmp_handle = fgds_cpp.fgds_file_handle(tmp_filename, True, device_id)
            del tmp_handle
        finally:
            if os.path.exists(tmp_filename):
                os.unlink(tmp_filename)
    except Exception as e:
        warnings.warn(
            f"FGDS is not available: {e}. Falling back to NoGDS.", UserWarning
        )
        return new_nogds_file_copier(
            device, bbuf_size_kb, max_threads, framework=framework, **kwargs
        )

    reader = fgds_cpp.fgds_file_reader(max_threads, device_id)

    def construct_copier(
        metadata: SafeTensorsMetadata, device: Device, framework: FrameworkOpBase
    ) -> CopierInterface:
        return FgdsFileCopier(
            metadata, device, reader, framework, max_threads=max_threads
        )

    return construct_copier
