# q_learning_board

## Francisco Teixeira Rocha Aragão - 2021031726

## Introdução

O presente trabalho impementa o algoritmo Q-Learning para aprender políticas (movimentações) dado um mapa de entrada.

## Organização do código

- main.py = Arquivo principal que chama as funções e executa o código.
- utils.py = Funções auxiliares para leitura da entrada.
- q_learning.py = Implementação do algoritmo Q-Learning e suas variações, além de gerir o mapa do jogo.

## Métodos

- standard: Implementação padrão do Q-Learning.
- positive: Implementação do Q-Learning com recompensas positivas.
- stochastic: Implementação do Q-Learning com movimentações estocásticas.

## Execução:

```bash
python main.py <arquivo_entrada> <metodo> <x_inicial> <y_inicial> <iteracoes> [--measure]

# metodo = standard, positive, stochastic.

# exemplo: python3 main.py input/choices.map stochastic 5 0 300000

# --measure = flag OPCIONAL para medir o tempo de execução.

```

## Dependências

- A única dependência externa utilizada foi o ```numpy==2.2.2``` para gerir listas e retornar a melhor opção de movimentação. Basta executar:

```bash
pip install numpy
```

ou então

```bash
pip install -r requirements.txt
```

para instalar a dependência.
