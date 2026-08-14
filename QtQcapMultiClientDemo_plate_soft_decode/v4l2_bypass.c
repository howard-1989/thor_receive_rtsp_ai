#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <unistd.h>

static int bypassed_fds[1024] = {0};

int v4l2_open(const char *pathname, int flags, ...) {
    int fd;
    int is_nv = (strstr(pathname, "nvdec") || strstr(pathname, "nvenc") || strstr(pathname, "msenc"));
    
    fprintf(stderr, "v4l2_bypass: v4l2_open(%s, %d) [is_nv=%d]\n", pathname, flags, is_nv);
    
    if (is_nv) {
        fd = open(pathname, flags);
        fprintf(stderr, "v4l2_bypass: bypassed to native open -> fd=%d\n", fd);
        if (fd >= 0 && fd < 1024) {
            bypassed_fds[fd] = 1;
        }
        return fd;
    }
    
    static int (*orig_v4l2_open)(const char *, int, ...) = NULL;
    if (!orig_v4l2_open) {
        orig_v4l2_open = dlsym(RTLD_NEXT, "v4l2_open");
    }
    
    // Handle optional mode argument
    mode_t mode = 0;
    if (flags & O_CREAT) {
        va_list ap;
        va_start(ap, flags);
        mode = va_arg(ap, mode_t);
        va_end(ap);
        return orig_v4l2_open(pathname, flags, mode);
    }
    return orig_v4l2_open(pathname, flags);
}

int v4l2_close(int fd) {
    fprintf(stderr, "v4l2_bypass: v4l2_close(%d)\n", fd);
    if (fd >= 0 && fd < 1024 && bypassed_fds[fd]) {
        bypassed_fds[fd] = 0;
        int ret = close(fd);
        fprintf(stderr, "v4l2_bypass: bypassed to native close -> %d\n", ret);
        return ret;
    }
    
    static int (*orig_v4l2_close)(int) = NULL;
    if (!orig_v4l2_close) {
        orig_v4l2_close = dlsym(RTLD_NEXT, "v4l2_close");
    }
    return orig_v4l2_close(fd);
}

int v4l2_ioctl(int fd, unsigned long request, ...) {
    va_list ap;
    va_start(ap, request);
    void *arg = va_arg(ap, void *);
    va_end(ap);

    fprintf(stderr, "v4l2_bypass: v4l2_ioctl(%d, 0x%lx) [bypassed=%d]\n", fd, request, (fd >= 0 && fd < 1024 && bypassed_fds[fd]));

    if (fd >= 0 && fd < 1024 && bypassed_fds[fd]) {
        return ioctl(fd, request, arg);
    }
    
    static int (*orig_v4l2_ioctl)(int, unsigned long, ...) = NULL;
    if (!orig_v4l2_ioctl) {
        orig_v4l2_ioctl = dlsym(RTLD_NEXT, "v4l2_ioctl");
    }
    return orig_v4l2_ioctl(fd, request, arg);
}
