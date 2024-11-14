# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing._data import StandardScaler
import scipy.cluster.hierarchy as shc
from io import BytesIO


# Funções auxiliares para conversão de DataFrame
@st.cache_data
def convert_df_to_csv(df):
    """Converte um DataFrame para CSV codificado em UTF-8."""
    return df.to_csv(index=False).encode('utf-8')


@st.cache_data
def convert_df_to_excel(df):
    """Converte um DataFrame para um arquivo Excel em Bytes, para download."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
        writer.save()
    return output.getvalue()


# Função de resumo do DataFrame
def summarize_df(df):
    """Exibe um resumo completo do DataFrame, incluindo informações gerais, contagem de valores únicos, ausentes e estatísticas."""
    st.write("### Resumo do DataFrame")
    st.write("Informações gerais:")
    st.write(df.info())
    st.write("\nContagem de valores únicos por coluna:")
    st.write(df.nunique())
    st.write("\nContagem de valores ausentes por coluna:")
    st.write(df.isna().sum())
    st.write("\nResumo estatístico:")
    st.write(df.describe())


# Função para realizar o upload de arquivo
def upload_file():
    """Realiza o upload do arquivo CSV ou XLS e retorna um DataFrame."""
    data_file = st.sidebar.file_uploader("Carregar arquivo CSV ou XLS", type=['csv', 'xlsx'])
    if data_file:
        if data_file.name.endswith('.csv'):
            df = pd.read_csv(data_file)
        elif data_file.name.endswith('.xlsx'):
            df = pd.read_excel(data_file)
        else:
            st.error("Tipo de arquivo não suportado!")
            return None
        return df
    else:
        return None


# Função principal para análise de agrupamento hierárquico
def hierarchical_clustering(df):
    """Realiza a análise de agrupamento hierárquico no DataFrame fornecido."""
    # Separar variáveis qualitativas e numéricas
    qualitative_vars = df.select_dtypes(include=['object']).columns
    numerical_vars = df.select_dtypes(include=['number']).columns

    # Codificar variáveis qualitativas
    df_encoded = pd.get_dummies(df, columns=qualitative_vars, drop_first=True)

    # Remover variáveis não utilizadas
    df_encoded = df_encoded.drop(columns=['OperatingSystems', 'Browser', 'Region', 'TrafficType', 'Revenue'],
                                 errors='ignore')

    # Padronização dos dados
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_encoded)

    # Calcular a matriz de ligação e plotar dendrogramas
    linkage_matrix = shc.linkage(df_scaled, method='ward')
    st.write("### Dendrograma para 3 grupos")
    plt.figure(figsize=(10, 6))
    shc.dendrogram(linkage_matrix, truncate_mode='lastp', p=3)
    st.pyplot(plt)

    st.write("### Dendrograma para 4 grupos")
    plt.figure(figsize=(10, 6))
    shc.dendrogram(linkage_matrix, truncate_mode='lastp', p=4)
    st.pyplot(plt)

    # Formar clusters e adicioná-los ao DataFrame
    clusters3 = shc.fcluster(linkage_matrix, t=169, criterion='distance')
    clusters4 = shc.fcluster(linkage_matrix, t=166, criterion='distance')
    df_encoded['cluster3'] = clusters3
    df_encoded['cluster4'] = clusters4

    # Visualizar a contagem dos clusters
    st.write("\nContagem de elementos em cada cluster (3 clusters):")
    st.write(df_encoded['cluster3'].value_counts())

    st.write("\nContagem de elementos em cada cluster (4 clusters):")
    st.write(df_encoded['cluster4'].value_counts())

    # Mesclar clusters no DataFrame original e retornar o resultado
    df = df.merge(df_encoded[['cluster3', 'cluster4']], on='id', how='left')
    return df, df_encoded


# Função para visualização dos clusters
def plot_clusters(df):
    """Gera visualizações para distribuição dos clusters e porcentagem de Revenue por cluster."""
    # Visualizar distribuição dos clusters
    st.write("### Distribuição dos Clusters - 3 grupos")
    sns.countplot(x='cluster3', data=df)
    plt.title('Distribuição dos Clusters - 3 grupos')
    st.pyplot(plt)

    st.write("### Distribuição dos Clusters - 4 grupos")
    sns.countplot(x='cluster4', data=df)
    plt.title('Distribuição dos Clusters - 4 grupos')
    st.pyplot(plt)


# Função principal da aplicação
def main():
    st.set_page_config(page_title="Análise de Agrupamento Hierárquico", layout="wide")
    st.title("Análise de Agrupamento Hierárquico - Segmentação de Clientes")

    # Upload do arquivo
    df = upload_file()
    if df is not None:
        st.write("### Visualização dos Dados Carregados")
        st.write(df.head())

        # Exibir resumo do DataFrame
        summarize_df(df)

        # Executar análise de agrupamento hierárquico
        df, df_encoded = hierarchical_clustering(df)

        # Exibir visualizações dos clusters
        plot_clusters(df)

        # Download do DataFrame final
        df_xlsx = convert_df_to_excel(df)
        st.download_button(label="📥 Download do Resultado", data=df_xlsx,
                           file_name="Resultado_Agrupamento_Hierarquico.xlsx")
        st.write("### Dados com Clusters")
        st.write(df.head())


if __name__ == "__main__":
    main()
