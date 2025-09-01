# Projeto de Manutenção Preditiva com Machine Learning
## Introdução
Este projeto desenvolve um sistema inteligente para manutenção preditiva de máquinas industriais, utilizando dados de sensores IoT. O objetivo principal é prever a ocorrência e o tipo de falha a partir de medições de sensores, gerando insights para otimizar as operações e evitar paradas não planejadas na produção.

## Metodologia
A metodologia adotada seguiu as melhores práticas da ciência de dados, abrangendo desde a limpeza inicial dos dados até a implantação do modelo.

## Preparação e Limpeza de Dados
- Identificação de Anomalias: Dados brutos foram analisados para identificar e tratar inconsistências. Valores como 'sim', 'não' e '-' foram mapeados para o formato binário (1 e 0).

- Tratamento de Outliers e Sinais: Foi adotada uma abordagem contextual para outliers. Valores fisicamente impossíveis (e.g., temperatura negativa em Kelvin) foram tratados como NaN, enquanto outliers válidos (e.g., velocidade_rotacional negativa) foram mantidos, pois indicavam estados operacionais específicos e valiosos para o modelo.

- Imputação e Codificação: Valores nulos em colunas numéricas foram imputados com a mediana. Variáveis categóricas (como o tipo de máquina) foram transformadas usando One-Hot Encoding para serem processadas pelos algoritmos.

## Estratégia de Modelagem
O problema foi abordado como uma tarefa de Classificação Multi-rótulo. A estratégia consistiu em treinar 5 modelos de forma independente, um para cada tipo de falha. A decisão de qual algoritmo usar para cada falha foi baseada em testes de desempenho.

Algoritmos Utilizados:

- RandomForestClassifier: Selecionado por sua alta precisão, robustez e capacidade de fornecer a importância das características.

- GradientBoostingClassifier: Escolhido por seu desempenho superior em algumas falhas, devido à sua abordagem de correção de erros sequencial.

- Otimização de Hiperparâmetros: Foi utilizado o RandomizedSearchCV para otimizar os hiperparâmetros dos modelos, buscando a melhor combinação de configurações para maximizar a performance.
