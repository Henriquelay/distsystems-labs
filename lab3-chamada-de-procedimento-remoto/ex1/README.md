# Dificuldados técnicas

Tive alguns problemas ao tentar seguir o roteiro no Arch.

O `rpcgen`, disponilizado no pacote `rpcsvc-proto` gerou um arquivo que importava que `rpc/rpc.h` que o meu sistema não o possuia. O que ele tinha era a `libtirpc` e o make gerado pelo `rpggen` incluia apenas a `libnsl`. A tirei e coloquei a flag de compilação `-I /usr/include/libtirpc`, e consertei os includes para `#include <tirpc/rpc/rpc.h>`

Consertado isso disso, iniciei o `rpcbind` com `sudo` e executei a aplicação normalmente
Achei bizarro o jeito que o RPCGEN gera para chavear as funções e o cast forçado para `char *` do meu resultado. Inicialmente resultava em valores errados pelo cast incorreto.
