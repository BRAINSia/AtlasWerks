/*
 * Copyright (c) 1986 - 1991 by Sun Microsystems, Inc.
 */

#ifndef _RPC_TYPES_H
#define	_RPC_TYPES_H


/*	types.h 1.23 88/10/25 SMI	*/

/*                                    
 * Rpc additions to <sys/types.h>
 */
/*
#include <sys/types.h>
*/

#ifdef __cplusplus
extern "C" {
#endif

typedef int bool_t;
typedef int enum_t;
typedef unsigned int u_int;
typedef unsigned long u_long;
typedef unsigned short u_short;
typedef unsigned char u_char;
typedef char * caddr_t;

/*
 * The following is part of a workaround for bug #1128007.
 * When it is fixed, this next typedef should be removed.
 */
typedef unsigned long u_ulonglong_t;

#define	__dontcare__	-1

#ifndef	FALSE
#define	FALSE	(0)
#endif

#ifndef	TRUE
#define	TRUE	(1)
#endif

#ifndef	NULL
#define	NULL	0
#endif

#ifndef	_KERNEL
#define	mem_alloc(bsize)	malloc(bsize)
#define	mem_free(ptr, bsize)	free(ptr)
#else
/*
#include <sys/kmem.h>
*/

#define	mem_alloc(bsize)	kmem_alloc((u_int)bsize, KM_SLEEP)
#define	mem_free(ptr, bsize)	kmem_free((caddr_t)(ptr), (u_int)(bsize))

#ifdef RPCDEBUG
#ifdef __STDC__
extern int	rpc_log(u_long, char *, int);
#else
extern int	rpc_log();
#endif
extern int	rpclog;

#define		RPCLOG(A, B, C) ((void)((rpclog) && rpc_log((A), (B), (C))))
#else
#define		RPCLOG(A, B, C)
#endif

#endif

#ifdef _NSL_RPC_ABI
/* For internal use only when building the libnsl RPC routines */
#define	select	_abi_select
#define	gettimeofday	_abi_gettimeofday
#define	syslog	_abi_syslog
#define	getgrent	_abi_getgrent
#define	endgrent	_abi_endgrent
#define	setgrent	_abi_setgrent
#endif

/* messaging stuff. */
#ifndef _KERNEL
#ifdef __STDC__
extern const char __nsl_dom[];
#else
extern char __nsl_dom[];
#endif
#endif

#ifdef __cplusplus
}
#endif

/*
#include <sys/time.h>
*/

#endif	/* _RPC_TYPES_H */
