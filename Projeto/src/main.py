import pandas as pd
from data_preparation import preparar_dados
from exploratory_analysis import realizar_analise_exploratoria
from model_training import treinar_modelos
from prediction_generation import gerar_predicoes_csv 
import joblib 
def main():
    """
    Orquestra o fluxo completo do projeto de manutenção preditiva:
    preparação de dados, análise exploratória, treinamento de modelos
    e geração de arquivo CSV com predições.
    """
    print("Iniciando o fluxo principal do projeto de manutenção preditiva...")

    # --- Configurações ---
    caminho_arquivo_dados = '/content/bootcamp_train.csv' 
    caminho_saida_predicoes = 'predictions_classes.csv' 

  
    print("\nExecutando etapa de Preparação e Limpeza dos Dados...")
    df_processado = preparar_dados(caminho_arquivo_dados)
    print("Etapa de Preparação e Limpeza dos Dados concluída.")

    

    print("\nExecutando etapa de Treinamento de Modelos...")
    # A função treinar_modelos retorna os pipelines, relatórios e previsões no conjunto de teste
    pipelines_treinados, relatorios_treinamento, predicoes_teste = treinar_modelos(df_processado)
    print("Etapa de Treinamento de Modelos concluída.")

  

    print("\nExecutando etapa de Geração de Arquivo CSV com Predições...")
    # Chama a função para gerar o CSV de predições, passando o DataFrame processado
   
    predictions_df_final = gerar_predicoes_csv(df_processado, pipelines_treinados, caminho_saida_csv=caminho_saida_predicoes)
    print(f"Etapa de Geração de Arquivo CSV com Predições concluída. Arquivo salvo em: {caminho_saida_predicoes}")

    print("\nFluxo principal do projeto concluído.")

if __name__ == "__main__":
    main()
