#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/eventfd.h>
#include <stdlib.h>
#include <sys/syscall.h>
#include <unistd.h>

/* Reserve low descriptors only for license descriptors. Never bypass FD_SET checks
 * or move unrelated descriptors above FD_SETSIZE. Load before starting QCAP. */
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static unsigned char reserved[1024], assigned[1024];
static int placeholder = -1;

__attribute__((constructor)) static void reserve_low_fds(void)
{
    int fd = syscall(SYS_openat, AT_FDCWD, "/dev/null", O_RDONLY | O_CLOEXEC, 0);
    if (fd < 0) return;
    placeholder = syscall(SYS_fcntl, fd, F_DUPFD_CLOEXEC, 1024);
    syscall(SYS_close, fd);
    if (placeholder < 0) return;
    int count = 0;
    for (; count < 64; ++count) {
        fd = syscall(SYS_fcntl, placeholder, F_DUPFD_CLOEXEC, 64);
        if (fd < 0) break;
        if (fd >= 1024) { syscall(SYS_close, fd); break; }
        reserved[fd] = 1;
    }
    if (getenv("QCAP_LOWFD_DEBUG"))
        fprintf(stderr, "[qcap-lowfd] pid=%ld reserved %d low FDs\n", (long)getpid(), count);
}

static int map_license_fd(int fd, void *caller, const char *kind)
{
    Dl_info info;
    int license = dladdr(caller, &info) &&
                  info.dli_fname && strstr(info.dli_fname, "libqcap2_lic.so");
    if (fd < 0 || !license) return fd;
    int flags = syscall(SYS_fcntl, fd, F_GETFD);
    pthread_mutex_lock(&lock);
    int low;
    for (low = 0; low < 1024; ++low)
        if (reserved[low] && !assigned[low]) break;
    if (low < 1024 && syscall(SYS_dup3, fd, low,
                              (flags & FD_CLOEXEC) ? O_CLOEXEC : 0) >= 0) {
        assigned[low] = 1;
        syscall(SYS_close, fd);
        if (getenv("QCAP_LOWFD_DEBUG"))
            fprintf(stderr, "[qcap-lowfd] License %s %d -> %d\n", kind, fd, low);
        fd = low;
    } else if (fd >= 1024) {
        syscall(SYS_close, fd);
        fd = -1;
        errno = EMFILE;
    }
    int saved_errno = errno;
    pthread_mutex_unlock(&lock);
    errno = saved_errno;
    return fd;
}

int socket(int domain, int type, int protocol)
{
    return map_license_fd(syscall(SYS_socket, domain, type, protocol),
                          __builtin_return_address(0), "socket");
}

int eventfd(unsigned int initval, int flags)
{
    return map_license_fd(syscall(SYS_eventfd2, initval, flags),
                          __builtin_return_address(0), "eventfd");
}

static int map_pipe(int pipefd[2], int flags, void *caller)
{
    if (syscall(SYS_pipe2, pipefd, flags) < 0) return -1;
    int first = map_license_fd(pipefd[0], caller, "pipe-read");
    if (first < 0) {
        int error = errno;
        close(pipefd[1]);
        errno = error;
        return -1;
    }
    int second = map_license_fd(pipefd[1], caller, "pipe-write");
    if (second < 0) {
        int error = errno;
        close(first);
        errno = error;
        return -1;
    }
    pipefd[0] = first;
    pipefd[1] = second;
    return 0;
}

int pipe(int pipefd[2])
{
    return map_pipe(pipefd, 0, __builtin_return_address(0));
}

int pipe2(int pipefd[2], int flags)
{
    return map_pipe(pipefd, flags, __builtin_return_address(0));
}

int close(int fd)
{
    if (fd < 0 || fd >= 1024) return syscall(SYS_close, fd);
    pthread_mutex_lock(&lock);
    int result;
    if (reserved[fd] && assigned[fd]) {
        /* Atomically replace the socket with a placeholder, keeping the slot. */
        result = syscall(SYS_dup3, placeholder, fd, O_CLOEXEC);
        if (result >= 0) { assigned[fd] = 0; result = 0; }
    } else {
        result = syscall(SYS_close, fd);
        if (result == 0) reserved[fd] = 0;
    }
    int saved_errno = errno;
    pthread_mutex_unlock(&lock);
    errno = saved_errno;
    return result;
}
