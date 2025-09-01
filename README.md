# Projeto IA — Sistema de Manutenção Preditiva em Máquinas Industriais
Este projeto aplica técnicas de aprendizado de máquina para construir um sistema inteligente de manutenção preditiva, capaz de identificar falhas em máquinas industriais com base em dados de sensores.

## Contexto do Projeto
O desafio foi desenvolver um sistema de controle de qualidade para um parque fabril, utilizando dados de dispositivos IoT para monitorar o comportamento das máquinas. O objetivo é prever a ocorrência e o tipo de falha, fornecendo a probabilidade associada, para otimizar a manutenção e evitar paradas inesperadas.

## Metodologia Utilizada
O projeto foi desenvolvido com uma abordagem robusta de machine learning supervisionado.

- Processo de Análise e Modelagem: Foi adotada uma estratégia de classificação multi-rótulo, treinando 5 modelos independentes, um para cada tipo de falha.

- Pipeline de Pré-processamento: Um pipeline automatizado com ColumnTransformer foi utilizado para imputar valores nulos, escalonar dados numéricos com MinMaxScaler e aplicar OneHotEncoder em variáveis categóricas.

- Seleção e Otimização: Os modelos RandomForestClassifier e GradientBoostingClassifier foram testados e otimizados com RandomizedSearchCV. A escolha final de cada modelo foi baseada na performance obtida para cada tipo de falha.

## Gestão de Atividades
A organização do projeto seguiu um fluxo de trabalho padrão da ciência de dados, com documentação clara em scripts Python:

- EDA (Exploração de Dados): Análise estatística e visualização de relações e outliers.

- Limpeza de Dados: Tratamento de valores inconsistentes e anomalias físicas (como temperaturas negativas em Kelvin).

- Modelagem e Tuning: Treinamento, otimização de hiperparâmetros e seleção do melhor algoritmo para cada tipo de falha.

- Geração de Predições: Criação e exportação de um arquivo final com as previsões.
