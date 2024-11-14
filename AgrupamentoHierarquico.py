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
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith(('.xls', '.xlsx')):
        return pd.read_excel(file)
    else:
        st.error("Tipo de arquivo não suportado.")
        return None

# Função para análise de cluster
def perform_clustering(df):
    # Definindo variáveis para o agrupamento
    qualitative_vars = df.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df, columns=qualitative_vars, drop_first=True)
    df_encoded = df_encoded.drop(columns=['OperatingSystems', 'Browser', 'Region', 'TrafficType', 'Revenue'], errors='ignore')

    # Padronização das variáveis
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_encoded)

    # Calculando a matriz de ligação e exibindo os dendrogramas
    linkage_matrix = shc.linkage(df_scaled, method='ward')
    plot_dendrogram(linkage_matrix, num_clusters=3)
    plot_dendrogram(linkage_matrix, num_clusters=4)

    # Criando os clusters
    clusters3 = shc.fcluster(linkage_matrix, t=169, criterion='distance')
    clusters4 = shc.fcluster(linkage_matrix, t=166, criterion='distance')
    df_encoded['cluster3'] = clusters3
    df_encoded['cluster4'] = clusters4

    # Mesclando clusters com o DataFrame original
    df['cluster3'] = clusters3
    df['cluster4'] = clusters4

    return df

# Função para exibir o dendrograma
def plot_dendrogram(linkage_matrix, num_clusters):
    plt.figure(figsize=(10, 6))
    plt.title(f"Dendrograma para {num_clusters} grupos")
    shc.dendrogram(linkage_matrix, truncate_mode='lastp', p=num_clusters)
    st.pyplot(plt.gcf())
    plt.clf()

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

        # Permite download do DataFrame com clusters
        download_button(data_clustered, filename="dados_clusterizados.csv")
else:
    st.warning("Por favor, faça o upload de um arquivo para iniciar a análise.")
