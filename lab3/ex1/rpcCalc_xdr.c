/*
 * Please do not edit this file.
 * It was generated using rpcgen.
 */

#include "rpcCalc.h"

bool_t
xdr_operands (XDR *xdrs, operands *objp)
{
	register int32_t *buf;

	 if (!xdr_int (xdrs, &objp->left))
		 return FALSE;
	 if (!xdr_int (xdrs, &objp->right))
		 return FALSE;
	return TRUE;
}
