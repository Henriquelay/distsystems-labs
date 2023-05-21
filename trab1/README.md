# Aprendizado Federado

Este trabalho foi desenvolvido por Henrique Coutinho Layber e Renan Moreira Gomes.

##  o código de Federated Learning

Este repositório contém o código para uma implementação básica de Federated Learning usando gRPC (Google Remote Procedure Call) em Python. Consiste em um servidor (server.py) e um cliente (client.py), que se comunicam por meio de mensagens definidas em um arquivo .proto.

## Pré-requisitos

Certifique-se de ter instalado os seguintes requisitos antes de executar o código:

- Python 3.x
- TensorFlow
- gRPC

## Executando o servidor

1. Abra um terminal e navegue até o diretório onde o arquivo server.py está localizado.
2. Execute o seguinte comando para iniciar o servidor:

   ```
   python server.py
   ```

   O servidor começará a ouvir em [::]:8080.

## Executando o cliente

1. Abra outro terminal e navegue até o diretório onde o arquivo client.py está localizado.
2. Execute o seguinte comando para iniciar o cliente:

   ```
   python client.py
   ```

   O cliente se conectará ao servidor no endereço 127.0.0.1:8080 por padrão. Se desejar usar um endereço ou porta diferentes, você pode especificá-los como argumentos de linha de comando:

   ```
   python client.py --server <endereço_do_servidor:porta> --address <endereço_do_cliente> --port <porta_do_cliente>
   ```

   Certifique-se de que o cliente e o servidor estejam sendo executados simultaneamente para estabelecer a comunicação correta.

## Metodologia no server.py

O servidor (FederatedLearningServer) implementa a lógica central do Federated Learning. Ele possui as seguintes características principais:

- O servidor espera a conexão de um número mínimo de clientes (definido por `min_clients`).
- Uma vez conectados os clientes suficientes, o servidor divide o conjunto de dados em lotes e os distribui entre os clientes para treinamento.
- O servidor aguarda até que todos os clientes terminem o treinamento e enviem seus pesos atualizados.
- Em seguida, o servidor calcula uma média ponderada dos pesos recebidos de cada cliente, usando o número de amostras correspondentes como peso.
- Os pesos médios são então distribuídos novamente para os clientes, que realizam a avaliação de seus modelos com base nesses pesos.
- Os resultados da avaliação são enviados de volta ao servidor, que verifica se a precisão média atingiu um limite pré-definido (`accuracy_threshold`).
- Se o limite de precisão for alcançado, o servidor encerra a execução. Caso contrário, ele continua para a próxima rodada de treinamento até atingir o número máximo de rodadas (`max_rounds`).

## APIs e Client Servicers implementados

### api

- **ClientRegister**: Esta API é usada pelos clientes para se registrarem no servidor. Eles enviam seu endereço IP, porta e ID do cliente. O servidor responde com um código de confirmação e o número da rodada atual.

### client

- **TrainingStart**: O cliente chama esta API para iniciar o treinamento. Ele fornece o índice inicial e final do lote de dados que ele deve usar, bem como o número de épocas de treinamento. O servidor responde com os pesos atuais do modelo e o número de amostras no lote.

- **ModelEvaluation**: Esta API é chamada pelo cliente para avaliar seu modelo usando os pesos fornecidos pelo servidor. O cliente retorna a precisão da avaliação para o servidor.


