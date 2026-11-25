"""Apply Linux restrictions before starting the caller's Python code."""

import ctypes
import errno
import resource
import sys


def restrict_process():
    for resource_type, limit in [
        (resource.RLIMIT_AS, 192 * 1024 * 1024),
        (resource.RLIMIT_FSIZE, 64 * 1024),
        (resource.RLIMIT_NOFILE, 32),
        (resource.RLIMIT_CORE, 0),
    ]:
        resource.setrlimit(resource_type, (limit, limit))

    library = ctypes.CDLL("libseccomp.so.2")
    library.seccomp_init.argtypes = [ctypes.c_uint32]
    library.seccomp_init.restype = ctypes.c_void_p
    library.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    library.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    library.seccomp_load.argtypes = [ctypes.c_void_p]
    library.seccomp_release.argtypes = [ctypes.c_void_p]
    context = library.seccomp_init(0x7FFF0000)  # SCMP_ACT_ALLOW
    if not context:
        raise RuntimeError("Could not initialize snippet isolation.")
    try:
        deny = 0x00050000 | errno.EPERM  # SCMP_ACT_ERRNO(EPERM)
        for name in (
            b"socket", b"socketpair", b"socketcall", b"clone", b"clone3", b"fork", b"vfork",
            b"execve", b"execveat", b"kill", b"tkill", b"tgkill", b"pidfd_send_signal",
            b"ptrace", b"process_vm_readv", b"process_vm_writev", b"pidfd_getfd",
            b"io_uring_setup", b"io_uring_enter", b"io_uring_register",
        ):
            syscall = library.seccomp_syscall_resolve_name(name)
            if syscall != -1 and library.seccomp_rule_add(context, deny, syscall, 0) != 0:
                raise RuntimeError("Could not configure snippet isolation.")
        if library.seccomp_load(context) != 0:
            raise RuntimeError("Could not enable snippet isolation.")
    finally:
        library.seccomp_release(context)


if __name__ == "__main__":
    code = compile(sys.stdin.buffer.read(), "<snippet>", "exec")
    restrict_process()
    exec(code, {"__name__": "__main__"})
