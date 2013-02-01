
/* brachy_ipc.h */

/*
 * This file contains definitions of interest in interprocess
 * communications between brachy processes.
 */

/* one-byte header codes for messages from the calculation engine */
#define C_BRACHY_MATRIX 'M'
#define C_BRACHY_VECTOR 'V'
#define C_BRACHY_ERROR 'E'
#define C_BRACHY_ACK 'A'
#define C_BRACHY_NAK 'N'
#define C_BRACHY_STAT 'S'

/* one-byte header codes for messages to the calculation engine */
#define C_BRACHY_GO 'S'
#define C_BRACHY_EXACT 'E'
#define C_BRACHY_ABORT 'X'
#define C_BRACHY_DESC 'D'
#define C_BRACHY_GRID 'G'
#define C_BRACHY_OBJECTS 'O'
#define C_BRACHY_POINTS 'P'
#define C_BRACHY_PATIENT 'W'
#define C_BRACHY_USER 'U'

#define SEND_CODE(a,b) {char foo; foo = b; sock_write (a, &foo, 1);}

/* Not really IPC stuff - but close - operating modes */
#define SIMPLE (1)
#define BACKGROUND (2)
#define JUST_A_DAEMON (3)
#define INETD_DAEMON (4)
