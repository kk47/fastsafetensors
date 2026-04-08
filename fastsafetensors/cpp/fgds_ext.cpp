// SPDX-License-Identifier: Apache-2.0

#ifdef _MSC_VER
// FGDS is Linux-only (depends on libfgds.so). On Windows we provide a minimal
// pybind11 stub module so the extension compiles cleanly; the Python side
// detects the missing classes and falls back to nogds gracefully.
#include <pybind11/pybind11.h>
namespace py = pybind11;
PYBIND11_MODULE(fgds_ext, m) {
    m.doc() = "FGDS extension module (stub: not available on Windows)";
    m.def("fgds_is_initialized", []() { return false; });
}
#else

#include <fcntl.h>
#include <cstring>
#include <unistd.h>
#include <sys/mman.h>
#include <chrono>
#include <dlfcn.h>
#include <cstdlib>
#include <algorithm>
#include <thread>
#include <mutex>
#include <vector>
#include <map>
#include <atomic>
#include <pybind11/pybind11.h>

#include "ext.hpp"

namespace py = pybind11;

#define ALIGN 65536

static bool debug_log = false;

void set_debug_log(bool _debug_log)
{
    debug_log = _debug_log;
}

typedef struct fgds_fileid {
    int fd;
    int device_id;
} fgds_fileid;

// FGDS library functions and state (similar to gds ext.cpp)
static void* fgds_lib_handle = nullptr;
static int (*fgds_open)(int) = nullptr;
static int (*fgds_close)(int) = nullptr;
static int (*fgds_regmem)(int, uintptr_t, size_t, void**) = nullptr;
static int (*fgds_deregmem)(int, uintptr_t, size_t) = nullptr;
static ssize_t (*fgds_read)(fgds_fileid, void*, size_t, off_t, size_t) = nullptr;
static std::atomic<int> fgds_ref_count{0};

template <typename T> void mydlsym(T** h, void* lib, const char* name) {
    *h = reinterpret_cast<T*>(dlsym(lib, name));
}

static void load_fgds_library() {
    int expected = 0;
    if (fgds_ref_count.compare_exchange_strong(expected, 1)) {
        void* lib = dlopen("libfgds.so", RTLD_LAZY | RTLD_GLOBAL | RTLD_NODELETE);
        if (!lib) {
            fgds_ref_count.store(0);
            throw std::runtime_error("Failed to load libfgds.so");
        }

        mydlsym(&fgds_open, lib, "fgds_open");
        mydlsym(&fgds_close, lib, "fgds_close");
        mydlsym(&fgds_regmem, lib, "fgds_regmem");
        mydlsym(&fgds_deregmem, lib, "fgds_deregmem");
        mydlsym(&fgds_read, lib, "fgds_read");
        if (!fgds_open || !fgds_close || !fgds_regmem ||
            !fgds_deregmem || !fgds_read) {
            dlclose(lib);
            fgds_ref_count.store(0);
            throw std::runtime_error("Failed to load FGDS functions");
        }
        fgds_lib_handle = lib;
    } else {
        fgds_ref_count.fetch_add(1);
    }
}

static void close_fgds_library() {
    if (fgds_ref_count.fetch_sub(1) == 1) {
        if (fgds_lib_handle) {
            dlclose(fgds_lib_handle);
            fgds_lib_handle = nullptr;
            fgds_open = nullptr;
            fgds_close = nullptr;
            fgds_regmem = nullptr;
            fgds_deregmem = nullptr;
            fgds_read = nullptr;
        }
    }
}

static bool is_fgds_initialized() {
    return fgds_lib_handle != nullptr;
}

// FGDS API wrapper
class FgdsWrapper {
public:
    FgdsWrapper() {
        load_fgds_library();
    }

    ~FgdsWrapper() {
        close_fgds_library();
    }

    int open(int device_id) {
        return fgds_open(device_id);
    }

    int close(int device_id) {
        return fgds_close(device_id);
    }

    int regmem(int device_id, uintptr_t addr, size_t size, void** target_addr) {
        return fgds_regmem(device_id, addr, size, target_addr);
    }

    int deregmem(int device_id, uintptr_t addr, size_t size) {
        return fgds_deregmem(device_id, addr, size);
    }

    ssize_t read(int fd, int device_id, void* buf, size_t length, off_t offset, size_t ptr_off) {
        fgds_fileid fid;
        fid.fd = fd;
        fid.device_id = device_id;
        return fgds_read(fid, buf, offset, length, ptr_off);
    }
};

// FGDS file handle class
class fgds_file_handle {
public:
    fgds_file_handle(std::string filename, bool o_direct, int device_id) {
        try {
            fgds_wrapper = new FgdsWrapper();
            this->device_id = device_id;
            this->filename = filename;

            // Open the file
            int flags = O_RDONLY;
            if (o_direct) {
                flags |= O_DIRECT;
            }
            fd = open(filename.c_str(), flags, 0644);
            if (fd < 0) {
                throw std::runtime_error("Failed to open file: " + filename);
            }

            // Initialize fileid
            fid.fd = fd;
            fid.device_id = device_id;
        } catch (const std::exception& e) {
            cleanup();
            throw;
        }
    }

    ~fgds_file_handle() {
        cleanup();
    }

    int get_device_id() const {
        return device_id;
    }

    int get_fd() const {
        return fd;
    }

    fgds_fileid get_fileid() const {
        return fid;
    }

    FgdsWrapper* get_wrapper() const {
        return fgds_wrapper;
    }

private:
    void cleanup() {
        if (fd >= 0) {
            close(fd);
            fd = -1;
        }
        if (fgds_wrapper) {
            delete fgds_wrapper;
            fgds_wrapper = nullptr;
        }
    }

    FgdsWrapper* fgds_wrapper = nullptr;
    int device_id = -1;
    int fd = -1;
    std::string filename;
    fgds_fileid fid;
};

// FGDS device buffer class
class fgds_device_buffer {
public:
    fgds_device_buffer(const uintptr_t dev_ptr, const uint64_t length) {
        _devPtr = dev_ptr;
        _length = length;
    }

    uintptr_t get_base_address() const {
        return _devPtr;
    }

    uint64_t get_length() const {
        return _length;
    }

private:
    uintptr_t _devPtr;
    uint64_t _length;
};

// FGDS file reader class
class fgds_file_reader {
public:
    fgds_file_reader(const int max_threads, int device_id) {
        _max_threads = max_threads;
        _device_id = device_id;
        _threads = new std::thread*[max_threads];
        for (int i = 0; i < max_threads; ++i) {
            _threads[i] = nullptr;
        }
        _next_id = 0;
    }

    ~fgds_file_reader() {
        if (_threads) {
            for (int i = 0; i < _max_threads; ++i) {
                if (_threads[i] != nullptr) {
                    _threads[i]->join();
                    delete _threads[i];
                }
            }
            delete[] _threads;
        }
    }

    const int submit_read(const fgds_file_handle& fh, const fgds_device_buffer& dst, const uint64_t offset, const uint64_t length, const uint64_t ptr_off) {
        int id = _next_id++;
        size_t thread_index = (size_t)(id % _max_threads);
        if (_threads[thread_index] != nullptr) {
            _threads[thread_index]->join();
            delete _threads[thread_index];
        }

        _threads[thread_index] = new std::thread(_thread, id, fh.get_fd(), fh.get_device_id(), fh.get_wrapper(), _device_id, dst.get_base_address(), length, offset, ptr_off, &_results, &_result_lock);

        return id;
    }

    const ssize_t wait_read(const int id) {
        size_t thread_index = (size_t)(id % _max_threads);
        if (_threads[thread_index] != nullptr) {
            _threads[thread_index]->join();
            delete _threads[thread_index];
            _threads[thread_index] = nullptr;
        }

        std::lock_guard<std::mutex> guard(_result_lock);
        ssize_t ret = _results[id];
        _results.erase(id);
        return ret;
    }

private:
    static void _thread(int thread_id, int fd, int device_id, FgdsWrapper* wrapper, int reader_device_id, uintptr_t dev_ptr, uint64_t length, uint64_t offset, uint64_t ptr_off, std::map<int, ssize_t>* results, std::mutex* result_lock) {
        ssize_t count = 0;
        void* devPtr_base = reinterpret_cast<void*>(dev_ptr);

        try {
            count = wrapper->read(fd, device_id, devPtr_base, length, offset, ptr_off);
            if (count < 0) {
                std::fprintf(stderr, "fgds_file_reader._thread: fgds_read returned an error: count=%zd\n", count);
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "fgds_file_reader._thread: exception: %s\n", e.what());
            count = -1;
        }

        std::lock_guard<std::mutex> guard(*result_lock);
        (*results)[thread_id] = count;
    }

    int _max_threads;
    int _device_id;
    std::thread** _threads;
    std::atomic<int> _next_id;
    std::map<int, ssize_t> _results;
    std::mutex _result_lock;
};

// Pybind11 bindings
PYBIND11_MODULE(fgds_ext, m) {
    m.doc() = "FGDS extension module";
    py::class_<fgds_file_handle>(m, "fgds_file_handle")
        .def(py::init<std::string, bool, int>())
        .def("get_device_id", &fgds_file_handle::get_device_id);

    py::class_<fgds_device_buffer>(m, "fgds_device_buffer")
        .def(py::init<const uintptr_t, const uint64_t>())
        .def("get_base_address", &fgds_device_buffer::get_base_address)
        .def("get_length", &fgds_device_buffer::get_length);

    py::class_<fgds_file_reader>(m, "fgds_file_reader")
        .def(py::init<const int, int>())
        .def("submit_read", &fgds_file_reader::submit_read)
        .def("wait_read", &fgds_file_reader::wait_read);

    // FGDS wrapper functions
    m.def("fgds_regmem", [](int device_id, uintptr_t addr, size_t size, pybind11::object target_addr_obj) {
        FgdsWrapper wrapper;
        void* target_addr = nullptr;
        int result = wrapper.regmem(device_id, addr, size, &target_addr);
        return result;
    });

    m.def("fgds_deregmem", [](int device_id, uintptr_t addr, size_t size) {
        FgdsWrapper wrapper;
        return wrapper.deregmem(device_id, addr, size);
    });

    m.def("fgds_read", [](int fd, int device_id, void* buf, size_t length, off_t offset, size_t ptr_off) {
        FgdsWrapper wrapper;
        return wrapper.read(fd, device_id, buf, length, offset, ptr_off);
    });

    // FGDS device open/close functions
    m.def("fgds_open", [](int device_id) {
        FgdsWrapper wrapper;
        return wrapper.open(device_id);
    });

    m.def("fgds_close", [](int device_id) {
        FgdsWrapper wrapper;
        return wrapper.close(device_id);
    });

    // Library management functions
    m.def("fgds_load_library", &load_fgds_library);
    m.def("fgds_close_library", &close_fgds_library);
    m.def("fgds_is_initialized", &is_fgds_initialized);
}
#endif
