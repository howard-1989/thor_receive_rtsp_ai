/**
 * select2poll.c — LD_PRELOAD wrapper
 *
 * 1. Shifts file descriptors < 1024 to >= 1024 for all non-license callers
 *    so that libqcap2_lic.so (which uses select) always gets FDs < 1024.
 * 2. Overrides select() → ppoll() to bypass FD_SETSIZE=1024 limit for other cases.
 * 3. Overrides __fdelt_chk/__fdelt_warn so FD_SET on fd>=1024 doesn't abort.
 *
 * Build:
 *   gcc -shared -fPIC -fno-omit-frame-pointer -funwind-tables -o libselect2poll.so select2poll.c -ldl
 *
 * Usage:
 *   LD_PRELOAD=./libselect2poll.so ./QtQcapMultiClientDemo_17kps
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <sys/select.h>
#include <poll.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <stdint.h>
#include <stdarg.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>
#include <sys/eventfd.h>
#include <sys/epoll.h>
#include <execinfo.h>

#ifndef O_TMPFILE
#define O_TMPFILE 020200000
#endif

/* ── Check if caller is libqcap2_lic.so ────────────────────────────────────── */
static int is_lic_caller(void)
{
    // 1. Direct caller check (extremely fast)
    void *ret_addr = __builtin_return_address(0);
    if (ret_addr) {
        Dl_info info;
        if (dladdr(ret_addr, &info) && info.dli_fname) {
            if (strstr(info.dli_fname, "libqcap2_lic.so")) {
                return 1;
            }
        }
    }

    // 2. Stack trace check (up to 8 frames back)
    void *buffer[8];
    int nptrs = backtrace(buffer, 8);
    for (int i = 1; i < nptrs; i++) {
        Dl_info info;
        if (dladdr(buffer[i], &info) && info.dli_fname) {
            if (strstr(info.dli_fname, "libqcap2_lic.so")) {
                return 1;
            }
        }
    }
    return 0;
}

/* ── Shift FD helper ──────────────────────────────────────────────────────── */
static int shift_fd(int fd)
{
    if (fd >= 0 && fd < 1024 && !is_lic_caller()) {
        int fd_flags = fcntl(fd, F_GETFD);
        int cmd = (fd_flags & FD_CLOEXEC) ? F_DUPFD_CLOEXEC : F_DUPFD;
        int new_fd = fcntl(fd, cmd, 1024);
        if (new_fd >= 0) {
            close(fd);
            return new_fd;
        }
    }
    return fd;
}

/* ── FD-creating System Call Hooks ────────────────────────────────────────── */

int socket(int domain, int type, int protocol)
{
    static int (*real_socket)(int, int, int) = NULL;
    if (!real_socket) real_socket = dlsym(RTLD_NEXT, "socket");
    return shift_fd(real_socket(domain, type, protocol));
}

int socketpair(int domain, int type, int protocol, int sv[2])
{
    static int (*real_socketpair)(int, int, int, int[2]) = NULL;
    if (!real_socketpair) real_socketpair = dlsym(RTLD_NEXT, "socketpair");
    int ret = real_socketpair(domain, type, protocol, sv);
    if (ret == 0) {
        sv[0] = shift_fd(sv[0]);
        sv[1] = shift_fd(sv[1]);
    }
    return ret;
}

int accept(int sockfd, struct sockaddr *addr, socklen_t *addrlen)
{
    static int (*real_accept)(int, struct sockaddr *, socklen_t *) = NULL;
    if (!real_accept) real_accept = dlsym(RTLD_NEXT, "accept");
    return shift_fd(real_accept(sockfd, addr, addrlen));
}

int accept4(int sockfd, struct sockaddr *addr, socklen_t *addrlen, int flags)
{
    static int (*real_accept4)(int, struct sockaddr *, socklen_t *, int) = NULL;
    if (!real_accept4) real_accept4 = dlsym(RTLD_NEXT, "accept4");
    return shift_fd(real_accept4(sockfd, addr, addrlen, flags));
}

int open(const char *pathname, int flags, ...)
{
    static int (*real_open)(const char *, int, ...) = NULL;
    if (!real_open) real_open = dlsym(RTLD_NEXT, "open");

    int needs_mode = (flags & O_CREAT);
    if ((flags & O_TMPFILE) == O_TMPFILE) {
        needs_mode = 1;
    }

    if (needs_mode) {
        va_list ap;
        va_start(ap, flags);
        mode_t mode = va_arg(ap, mode_t);
        va_end(ap);
        return shift_fd(real_open(pathname, flags, mode));
    }
    return shift_fd(real_open(pathname, flags));
}

int open64(const char *pathname, int flags, ...)
{
    static int (*real_open64)(const char *, int, ...) = NULL;
    if (!real_open64) real_open64 = dlsym(RTLD_NEXT, "open64");

    int needs_mode = (flags & O_CREAT);
    if ((flags & O_TMPFILE) == O_TMPFILE) {
        needs_mode = 1;
    }

    if (needs_mode) {
        va_list ap;
        va_start(ap, flags);
        mode_t mode = va_arg(ap, mode_t);
        va_end(ap);
        return shift_fd(real_open64(pathname, flags, mode));
    }
    return shift_fd(real_open64(pathname, flags));
}

int openat(int dirfd, const char *pathname, int flags, ...)
{
    static int (*real_openat)(int, const char *, int, ...) = NULL;
    if (!real_openat) real_openat = dlsym(RTLD_NEXT, "openat");

    int needs_mode = (flags & O_CREAT);
    if ((flags & O_TMPFILE) == O_TMPFILE) {
        needs_mode = 1;
    }

    if (needs_mode) {
        va_list ap;
        va_start(ap, flags);
        mode_t mode = va_arg(ap, mode_t);
        va_end(ap);
        return shift_fd(real_openat(dirfd, pathname, flags, mode));
    }
    return shift_fd(real_openat(dirfd, pathname, flags));
}

int openat64(int dirfd, const char *pathname, int flags, ...)
{
    static int (*real_openat64)(int, const char *, int, ...) = NULL;
    if (!real_openat64) real_openat64 = dlsym(RTLD_NEXT, "openat64");

    int needs_mode = (flags & O_CREAT);
    if ((flags & O_TMPFILE) == O_TMPFILE) {
        needs_mode = 1;
    }

    if (needs_mode) {
        va_list ap;
        va_start(ap, flags);
        mode_t mode = va_arg(ap, mode_t);
        va_end(ap);
        return shift_fd(real_openat64(dirfd, pathname, flags, mode));
    }
    return shift_fd(real_openat64(dirfd, pathname, flags));
}

int creat(const char *pathname, mode_t mode)
{
    static int (*real_creat)(const char *, mode_t) = NULL;
    if (!real_creat) real_creat = dlsym(RTLD_NEXT, "creat");
    return shift_fd(real_creat(pathname, mode));
}

int creat64(const char *pathname, mode_t mode)
{
    static int (*real_creat64)(const char *, mode_t) = NULL;
    if (!real_creat64) real_creat64 = dlsym(RTLD_NEXT, "creat64");
    return shift_fd(real_creat64(pathname, mode));
}

int pipe(int pipefd[2])
{
    static int (*real_pipe)(int[2]) = NULL;
    if (!real_pipe) real_pipe = dlsym(RTLD_NEXT, "pipe");
    int ret = real_pipe(pipefd);
    if (ret == 0) {
        pipefd[0] = shift_fd(pipefd[0]);
        pipefd[1] = shift_fd(pipefd[1]);
    }
    return ret;
}

int pipe2(int pipefd[2], int flags)
{
    static int (*real_pipe2)(int[2], int) = NULL;
    if (!real_pipe2) real_pipe2 = dlsym(RTLD_NEXT, "pipe2");
    int ret = real_pipe2(pipefd, flags);
    if (ret == 0) {
        pipefd[0] = shift_fd(pipefd[0]);
        pipefd[1] = shift_fd(pipefd[1]);
    }
    return ret;
}

int eventfd(unsigned int initval, int flags)
{
    static int (*real_eventfd)(unsigned int, int) = NULL;
    if (!real_eventfd) real_eventfd = dlsym(RTLD_NEXT, "eventfd");
    return shift_fd(real_eventfd(initval, flags));
}

int epoll_create(int size)
{
    static int (*real_epoll_create)(int) = NULL;
    if (!real_epoll_create) real_epoll_create = dlsym(RTLD_NEXT, "epoll_create");
    return shift_fd(real_epoll_create(size));
}

int epoll_create1(int flags)
{
    static int (*real_epoll_create1)(int) = NULL;
    if (!real_epoll_create1) real_epoll_create1 = dlsym(RTLD_NEXT, "epoll_create1");
    return shift_fd(real_epoll_create1(flags));
}

int dup(int oldfd)
{
    static int (*real_dup)(int) = NULL;
    if (!real_dup) real_dup = dlsym(RTLD_NEXT, "dup");
    return shift_fd(real_dup(oldfd));
}


/* ── Bypass glibc fortified FD_SET/FD_CLR check ────────────────────────── */
/* Suppress "bit out of range" abort. The real select() is also replaced,
   so large FDs will be handled via ppoll() which has no 1024 limit. */
long __fdelt_chk(long d)
{
    return d / (8 * (long)sizeof(long));
}

long __fdelt_warn(long d)
{
    return d / (8 * (long)sizeof(long));
}

/* ── select() → ppoll() wrapper ────────────────────────────────────────── */
static int (*real_select)(int, fd_set *, fd_set *, fd_set *, struct timeval *) = NULL;

static struct pollfd *fdset_to_pollfds(int nfds, fd_set *fds, short events, int *count_out)
{
    if (!fds) { *count_out = 0; return NULL; }
    int cnt = 0;
    for (int i = 0; i < nfds; i++) {
        if (FD_ISSET(i, fds)) cnt++;
    }
    if (cnt == 0) { *count_out = 0; return NULL; }

    struct pollfd *pfds = (struct pollfd *)malloc(cnt * sizeof(struct pollfd));
    if (!pfds) { *count_out = 0; return NULL; }

    int idx = 0;
    for (int i = 0; i < nfds; i++) {
        if (FD_ISSET(i, fds)) {
            pfds[idx].fd = i;
            pfds[idx].events = events;
            pfds[idx].revents = 0;
            idx++;
        }
    }
    *count_out = cnt;
    return pfds;
}

static void pollfds_to_fdset(struct pollfd *pfds, int count, fd_set *fds, short events)
{
    if (!fds) return;
    FD_ZERO(fds);
    for (int i = 0; i < count; i++) {
        if (pfds[i].revents & (events | POLLERR | POLLHUP | POLLNVAL)) {
            FD_SET(pfds[i].fd, fds);
        }
    }
}

int select(int nfds, fd_set *readfds, fd_set *writefds,
           fd_set *exceptfds, struct timeval *timeout)
{
    if (!real_select) {
        real_select = (int (*)(int, fd_set *, fd_set *, fd_set *, struct timeval *))
                      dlsym(RTLD_NEXT, "select");
    }

    /* If within FD_SETSIZE range, just use real select() */
    if (nfds <= FD_SETSIZE && real_select) {
        return real_select(nfds, readfds, writefds, exceptfds, timeout);
    }

    /* Convert to ppoll() */

    /* Build pollfd array from readfds + writefds */
    struct pollfd *all_pfds = NULL;
    int total_count = 0;

    struct pollfd *r = fdset_to_pollfds(nfds, readfds, POLLIN, &total_count);
    all_pfds = r;
    int r_cnt = total_count;

    struct pollfd *w = fdset_to_pollfds(nfds, writefds, POLLOUT, &total_count);
    if (w) {
        all_pfds = (struct pollfd *)realloc(all_pfds, (r_cnt + total_count) * sizeof(struct pollfd));
        if (!all_pfds) { free(r); free(w); errno = ENOMEM; return -1; }
        memcpy(all_pfds + r_cnt, w, total_count * sizeof(struct pollfd));
        int w_cnt = total_count;
        total_count = r_cnt + w_cnt;
        free(w);
    } else {
        total_count = r_cnt;
    }

    /* Also add exceptfds (rarely used) */
    struct pollfd *e = fdset_to_pollfds(nfds, exceptfds, POLLPRI, &total_count);
    if (e) {
        int old_cnt = total_count;
        all_pfds = (struct pollfd *)realloc(all_pfds, (old_cnt + total_count) * sizeof(struct pollfd));
        if (!all_pfds) { free(r); free(e); errno = ENOMEM; return -1; }
        memcpy(all_pfds + old_cnt, e, total_count * sizeof(struct pollfd));
        total_count = old_cnt + total_count;
        free(e);
    }

    /* Convert timeout */
    struct timespec ts, *pts = NULL;
    if (timeout) {
        ts.tv_sec = timeout->tv_sec;
        ts.tv_nsec = timeout->tv_usec * 1000;
        pts = &ts;
    }

    int ret = ppoll(all_pfds, (nfds_t)(total_count > 0 ? total_count : 1), pts, NULL);

    if (ret > 0) {
        pollfds_to_fdset(all_pfds, total_count, readfds, POLLIN);
        if (writefds) {
            FD_ZERO(writefds);
            for (int i = 0; i < total_count; i++) {
                if (all_pfds[i].revents & (POLLOUT | POLLERR | POLLHUP))
                    FD_SET(all_pfds[i].fd, writefds);
            }
        }
        if (exceptfds) {
            FD_ZERO(exceptfds);
            for (int i = 0; i < total_count; i++) {
                if (all_pfds[i].revents & (POLLPRI | POLLERR | POLLHUP))
                    FD_SET(all_pfds[i].fd, exceptfds);
            }
        }
    } else if (ret == 0) {
        /* timeout — all sets empty */
        if (readfds) FD_ZERO(readfds);
        if (writefds) FD_ZERO(writefds);
        if (exceptfds) FD_ZERO(exceptfds);
    }

    free(all_pfds);
    return ret;
}
