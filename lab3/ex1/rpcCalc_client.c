/*
 * This is sample code generated by rpcgen.
 * These are only templates and you can use them
 * as a guideline for developing your own functions.
 */

#include "rpcCalc.h"
#include <stdio.h>
#include <stdlib.h>

void prog_100(char *host)
{
	CLIENT *clnt;
	int *result_1;
	operands add_100_arg;
	int *result_2;
	operands sub_100_arg;

	int left, right;
	printf("Input numbers to calc: ");
	scanf("%d %d", &left, &right);
	printf("Scanned: %d, %d\n", left, right);
	add_100_arg.left = left;
	add_100_arg.right = right;
	sub_100_arg.left = left;
	sub_100_arg.right = right;

#ifndef DEBUG
	clnt = clnt_create(host, PROG, VER, "udp");
	if (clnt == NULL)
	{
		clnt_pcreateerror(host);
		exit(1);
	}
#endif /* DEBUG */

	result_1 = add_100(&add_100_arg, clnt);
	if (result_1 == (int *)NULL)
	{
		clnt_perror(clnt, "call failed");
	}
	result_2 = sub_100(&sub_100_arg, clnt);
	if (result_2 == (int *)NULL)
	{
		clnt_perror(clnt, "call failed");
	}

	printf("Add: %d\nSub: %d\n", result_1, result_2);

#ifndef DEBUG
	clnt_destroy(clnt);
#endif /* DEBUG */
}

int main(int argc, char *argv[])
{
	char *host;

	if (argc < 2)
	{
		printf("usage: %s server_host\n", argv[0]);
		exit(1);
	}
	host = argv[1];
	prog_100(host);
	exit(0);
}
