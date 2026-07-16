#define _GNU_SOURCE
#include <dlfcn.h>
#include <sys/select.h>
#include <poll.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdarg.h>
#include <sys/types.h>
#include <sys/stat.h>

/* ── Bypass glibc fortified FD_SET/FD_CLR check ────────────────────────── */
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

    if (nfds <= FD_SETSIZE && real_select) {
        return real_select(nfds, readfds, writefds, exceptfds, timeout);
    }

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

    struct pollfd *e = fdset_to_pollfds(nfds, exceptfds, POLLPRI, &total_count);
    if (e) {
        int old_cnt = total_count;
        all_pfds = (struct pollfd *)realloc(all_pfds, (old_cnt + total_count) * sizeof(struct pollfd));
        if (!all_pfds) { free(r); free(e); errno = ENOMEM; return -1; }
        memcpy(all_pfds + old_cnt, e, total_count * sizeof(struct pollfd));
        total_count = old_cnt + total_count;
        free(e);
    }

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
        if (readfds) FD_ZERO(readfds);
        if (writefds) FD_ZERO(writefds);
        if (exceptfds) FD_ZERO(exceptfds);
    }

    free(all_pfds);
    return ret;
}

/* ── File Descriptor Redirection to keep socket FDs < 1000 ──────────────── */
static int (*real_open)(const char *, int, ...) = NULL;
static int (*real_open64)(const char *, int, ...) = NULL;
static int (*real_openat)(int, const char *, int, ...) = NULL;
static int (*real_openat64)(int, const char *, int, ...) = NULL;
static int (*real_creat)(const char *, mode_t) = NULL;
static int (*real_creat64)(const char *, mode_t) = NULL;
static int (*real_dup)(int) = NULL;
static int (*real_dup2)(int, int) = NULL;
static int (*real_dup3)(int, int, int) = NULL;
static int (*real_fcntl)(int, int, ...) = NULL;

__attribute__((constructor)) static void init_all() {
    real_select = dlsym(RTLD_NEXT, "select");
    real_open = dlsym(RTLD_NEXT, "open");
    real_open64 = dlsym(RTLD_NEXT, "open64");
    real_openat = dlsym(RTLD_NEXT, "openat");
    real_openat64 = dlsym(RTLD_NEXT, "openat64");
    real_creat = dlsym(RTLD_NEXT, "creat");
    real_creat64 = dlsym(RTLD_NEXT, "creat64");
    real_dup = dlsym(RTLD_NEXT, "dup");
    real_dup2 = dlsym(RTLD_NEXT, "dup2");
    real_dup3 = dlsym(RTLD_NEXT, "dup3");
    real_fcntl = dlsym(RTLD_NEXT, "fcntl");
}

static int redirect_fd(int fd) {
    if (fd >= 0 && fd < 1000) {
        int new_fd = fcntl(fd, F_DUPFD, 1000);
        if (new_fd >= 0) {
            close(fd);
            return new_fd;
        }
    }
    return fd;
}

int open(const char *pathname, int flags, ...) {
    mode_t mode = 0;
    if (flags & O_CREAT) {
        va_list args;
        va_start(args, flags);
        mode = va_arg(args, mode_t);
        va_end(args);
    }
    int fd = real_open ? real_open(pathname, flags, mode) : -1;
    return redirect_fd(fd);
}

int open64(const char *pathname, int flags, ...) {
    mode_t mode = 0;
    if (flags & O_CREAT) {
        va_list args;
        va_start(args, flags);
        mode = va_arg(args, mode_t);
        va_end(args);
    }
    int fd = real_open64 ? real_open64(pathname, flags, mode) : (real_open ? real_open(pathname, flags, mode) : -1);
    return redirect_fd(fd);
}

int openat(int dirfd, const char *pathname, int flags, ...) {
    mode_t mode = 0;
    if (flags & O_CREAT) {
        va_list args;
        va_start(args, flags);
        mode = va_arg(args, mode_t);
        va_end(args);
    }
    int fd = real_openat ? real_openat(dirfd, pathname, flags, mode) : -1;
    return redirect_fd(fd);
}

int openat64(int dirfd, const char *pathname, int flags, ...) {
    mode_t mode = 0;
    if (flags & O_CREAT) {
        va_list args;
        va_start(args, flags);
        mode = va_arg(args, mode_t);
        va_end(args);
    }
    int fd = real_openat64 ? real_openat64(dirfd, pathname, flags, mode) : (real_openat ? real_openat(dirfd, pathname, flags, mode) : -1);
    return redirect_fd(fd);
}

int creat(const char *pathname, mode_t mode) {
    int fd = real_creat ? real_creat(pathname, mode) : -1;
    return redirect_fd(fd);
}

int creat64(const char *pathname, mode_t mode) {
    int fd = real_creat64 ? real_creat64(pathname, mode) : (real_creat ? real_creat(pathname, mode) : -1);
    return redirect_fd(fd);
}

int dup(int oldfd) {
    int fd = real_dup ? real_dup(oldfd) : -1;
    if (oldfd >= 1000) {
        return redirect_fd(fd);
    }
    return fd;
}

int dup2(int oldfd, int newfd) {
    return real_dup2 ? real_dup2(oldfd, newfd) : -1;
}

int dup3(int oldfd, int newfd, int flags) {
    return real_dup3 ? real_dup3(oldfd, newfd, flags) : -1;
}

int fcntl(int fd, int cmd, ...) {
    va_list args;
    va_start(args, cmd);
    void *arg = va_arg(args, void *);
    va_end(args);

    if (!real_fcntl) return -1;

    if (cmd == F_DUPFD || cmd == F_DUPFD_CLOEXEC) {
        long target = (long)arg;
        if (fd >= 1000 && target < 1000) {
            target = 1000;
        }
        return real_fcntl(fd, cmd, target);
    }

    return real_fcntl(fd, cmd, arg);
}
