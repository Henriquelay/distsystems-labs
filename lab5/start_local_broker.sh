#!/bin/bash

# Verifica se o Mosquitto está instalado
if ! command -v mosquitto &> /dev/null; then
    echo "O Mosquitto não está instalado. Por favor, instale-o antes de executar este script."
    exit 1
fi

# Inicia o serviço do Mosquitto
sudo service mosquitto start

# Verifica se o serviço foi iniciado corretamente
if [ $? -eq 0 ]; then
    echo "Broker MQTT local iniciado com sucesso."
else
    echo "Falha ao iniciar o broker MQTT local."
fi