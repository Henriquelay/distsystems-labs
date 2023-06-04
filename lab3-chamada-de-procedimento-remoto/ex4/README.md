# Exercicio 4

Esse é um servidor/cliente de uma blockchain simples que permite a mineração de novos blocos. O cliente permite ao usuário se comunicar com o servidor para obter informações sobre a transação atual, como o ID da transação, o desafio a ser resolvido e se há um vencedor para a transação. O cliente também permite que o usuário submeta uma solução para o desafio e receba uma resposta sobre se a solução está correta e, em caso afirmativo, se o usuário é o vencedor da transação.

O servidor mantém uma lista de transações e cria um novo objeto de transação sempre que um cliente solicita um ID de transação. Cada objeto de transação contém um desafio e uma solução, e é atribuído um vencedor quando um cliente submete uma solução correta para o desafio. O servidor é implementado usando o protocolo gRPC, uma tecnologia de comunicação remota que usa o formato protobuf para serialização de dados.

A solução do desafio é gerada pela função create_solution(), que utiliza a biblioteca hashlib para calcular o SHA1 da soma de dois números aleatórios gerados entre 1 e 100000 multiplicados pelo valor do desafio. O resultado é retornado como uma string hexadecimal.


## Instruções de execução

Para executar o servidor, basta rodar o comando python3 server.py. Para executar o cliente, é necessário passar o endereço do servidor com a flag --server no comando python3 client.py. Se o servidor estiver rodando localmente na porta 50051, pode-se executar o cliente sem especificar o endereço do servidor.

Para executar o servidor, execute o seguinte comando:

```bash
python3 server.py
```

Para executar o cliente, execute o seguinte comando:

```bash
python3 client.py --server <server_address>
```

ou se você estiver executando o servidor em um host local na porta 50051:

```bash
python3 client.py
```
