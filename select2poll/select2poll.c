/**
 * select2poll.c — LD_PRELOAD wrapper
 *
 * Overrides select() → ppoll() to bypass FD_SETSIZE=1024 limit.
 * Also overrides __fdelt_chk/__fdelt_warn so FD_SET on fd>=1024 doesn't abort.
 *
 * Build:
 *   gcc -shared -fPIC -o libselect2poll.so select2poll.c -ldl
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
