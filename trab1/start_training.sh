#!/bin/bash

# Iniciar o servidor no 1x1
tmux new-session -d -s grid

tmux select-pane -t 0
tmux send-keys "poetry run python3 server.py" C-m

# Criar os demais painéis
tmux split-window -h
tmux split-window -h
tmux split-window -v
tmux split-window -v

# Selecionar e executar o cliente em cada painel
tmux select-pane -t 1
tmux send-keys "poetry run python3 client.py --port=50051" C-m

tmux select-pane -t 2
tmux send-keys "poetry run python3 client.py --port=50052" C-m

tmux select-pane -t 3
tmux send-keys "poetry run python3 client.py --port=50053" C-m

tmux select-pane -t 4
tmux send-keys "poetry run python3 client.py --port=50054" C-m



# Redimensionar painéis para igualar tamanho
tmux select-layout even-horizontal

# Anexar à sessão do tmux
tmux attach-session -t grid
