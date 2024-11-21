import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as shc
from io import StringIO

# Título e descrição do aplicativo
st.title("Clusterização de Dados de Intenção de Compras Online")

# Barra lateral para upload do arquivo
st.sidebar.header("Upload do Arquivo")
uploaded_file = st.sidebar.file_uploader("Selecione um arquivo CSV ou XLS", type=["csv", "xls", "xlsx"])

# Função para carregar o arquivo
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        elif file.name.endswith(('.xls', '.xlsx')):
            return pd.read_excel(file)
        else:
            st.error("Tipo de arquivo não suportado.")
            return None
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo: {e}")
        return None

# Função para análise de cluster
def perform_clustering(df):
    try:
        # Definindo variáveis para o agrupamento
        qualitative_vars = df.select_dtypes(include=['object']).columns
        df_encoded = pd.get_dummies(df, columns=qualitative_vars, drop_first=True)
        df_encoded = df_encoded.drop(columns=['OperatingSystems', 'Browser', 'Region', 'TrafficType', 'Revenue'], errors='ignore')

        # Padronização das variáveis
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_encoded)

        # Calculando a matriz de ligação
        linkage_matrix = shc.linkage(df_scaled, method='ward')

        # Exibindo os dendrogramas
        st.subheader("Dendrogramas")
        plot_dendrogram(linkage_matrix, num_clusters=3)
        plot_dendrogram(linkage_matrix, num_clusters=4)

        # Criando os clusters
        clusters3 = shc.fcluster(linkage_matrix, t=3, criterion='maxclust')
        clusters4 = shc.fcluster(linkage_matrix, t=4, criterion='maxclust')
        df['cluster3'] = clusters3
        df['cluster4'] = clusters4

        st.success("Clusterização realizada com sucesso!")
        return df
    except Exception as e:
        st.error(f"Erro durante a clusterização: {e}")
        return None

# Função para exibir o dendrograma
def plot_dendrogram(linkage_matrix, num_clusters):
    try:
        plt.figure(figsize=(10, 6))
        plt.title(f"Dendrograma para {num_clusters} grupos")
        shc.dendrogram(linkage_matrix, truncate_mode='lastp', p=num_clusters)
        st.pyplot(plt.gcf())
        plt.close()
    except Exception as e:
        st.error(f"Erro ao gerar o dendrograma: {e}")

# Função para permitir download do DataFrame
def download_button(dataframe, filename="data.csv"):
    csv = dataframe.to_csv(index=False)
    st.download_button(
        label="Baixar Dados Clusterizados",
        data=csv,
        file_name=filename,
        mime='text/csv',
    )

# Processamento do arquivo
if uploaded_file:
    data = load_data(uploaded_file)
    if data is not None:
        st.subheader("Dados Carregados")
        st.write(data.head())

        # Realiza a análise de cluster
        data_clustered = perform_clustering(data)

        if data_clustered is not None:
            # Exibe os clusters criados
            st.subheader("Dados com Clusters")
            st.write(data_clustered.head())

            # Permite download do DataFrame com clusters
            download_button(data_clustered, filename="dados_clusterizados.csv")
else:
    st.warning("Por favor, faça o upload de um arquivo para iniciar a análise.")

