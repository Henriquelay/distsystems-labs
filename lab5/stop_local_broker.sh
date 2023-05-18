#!/bin/bash

# Para o serviço do Mosquitto
sudo service mosquitto stop

# Verifica se o serviço foi interrompido corretamente
if [ $? -eq 0 ]; then
    echo "Broker MQTT local foi interrompido com sucesso."
else
    echo "Falha ao interromper o broker MQTT local."
fi