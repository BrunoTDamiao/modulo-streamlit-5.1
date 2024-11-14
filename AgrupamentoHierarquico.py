import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as shc

# Configuração de estilo
st.set_option('deprecation.showPyplotGlobalUse', False)
sns.set(style="whitegrid")

# Título principal do app
st.title("Análise de Clusters em Dados de Intenção de Compras Online")

# Barra lateral para upload do arquivo
st.sidebar.header("Upload do Arquivo")
uploaded_file = st.sidebar.file_uploader("Selecione um arquivo CSV ou XLS", type=["csv", "xls", "xlsx"])

# Função para carregar o dataset com base no tipo de arquivo
if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    df.index.name = 'id'
else:
    st.warning("Por favor, faça o upload de um arquivo para iniciar a análise.")
    st.stop()  # Interrompe o código até que um arquivo seja carregado

# Exibindo o dataset inicial
st.subheader("Primeiras linhas do DataFrame")
st.write(df.head())

# Exibindo a contagem de Revenue
st.subheader("Contagem de Revenue")
st.write(df['Revenue'].value_counts(dropna=False))

# Função para exibir resumo completo do DataFrame
def summarize_df(df):
    st.write("### Informações gerais:")
    buffer = st.empty()
    df.info(buf=buffer)  # buffer for in-line display
    st.text(buffer)
    
    st.write("### Contagem de valores únicos por coluna:")
    st.write(df.nunique())
    
    st.write("### Contagem de valores ausentes por coluna:")
    st.write(df.isna().sum())
    
    st.write("### Resumo estatístico:")
    st.write(df.describe())

summarize_df(df)

# Separando variáveis qualitativas e numéricas
qualitative_vars = df.select_dtypes(include=['object']).columns
numerical_vars = df.select_dtypes(include=['number']).columns

st.write("Variáveis qualitativas:", qualitative_vars)
st.write("Variáveis numéricas:", numerical_vars)

# Codificação de variáveis qualitativas
df_encoded = pd.get_dummies(df, columns=qualitative_vars, drop_first=True)
df_encoded = df_encoded.drop(columns=['OperatingSystems', 'Browser', 'Region', 'TrafficType', 'Revenue'])

# Padronização
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_encoded)

# Dendrograma para 3 e 4 grupos
st.subheader("Dendrogramas para 3 e 4 grupos")
linkage_matrix = shc.linkage(df_scaled, method='ward')

fig, ax = plt.subplots(figsize=(10, 6))
plt.title("Dendrograma para 3 grupos")
shc.dendrogram(linkage_matrix, truncate_mode='lastp', p=3)
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(10, 6))
plt.title("Dendrograma para 4 grupos")
shc.dendrogram(linkage_matrix, truncate_mode='lastp', p=4)
st.pyplot(fig)

# Clusterização e contagem dos elementos em cada cluster
clusters3 = shc.fcluster(linkage_matrix, t=169, criterion='distance')
df_encoded['cluster3'] = clusters3

clusters4 = shc.fcluster(linkage_matrix, t=166, criterion='distance')
df_encoded['cluster4'] = clusters4

st.write("Contagem de elementos em cada cluster, para 3 clusters:", df_encoded['cluster3'].value_counts())
st.write("Contagem de elementos em cada cluster, para 4 clusters:", df_encoded['cluster4'].value_counts())

# Gráficos de distribuição dos clusters
st.subheader("Distribuição dos Clusters")
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x='cluster3', data=df_encoded, ax=ax)
plt.title('Distribuição dos Clusters - 3 grupos')
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x='cluster4', data=df_encoded, ax=ax)
plt.title('Distribuição dos Clusters - 4 grupos')
st.pyplot(fig)

# Merge dos clusters com o DataFrame original
df = df.merge(df_encoded[['cluster3', 'cluster4']], on='id', how='left')

# Contagem de Revenue 0 e 1 por Cluster
st.subheader("Contagem de Revenue 0 e 1 por Cluster - 3 e 4 grupos")
count_revenue3 = df.groupby(['cluster3', 'Revenue']).size().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(6, 4))
count_revenue3.plot(kind='bar', stacked=True, ax=ax)
plt.title('Contagem de Revenue 0 e 1 por Cluster - 3 grupos')
st.pyplot(fig)

count_revenue4 = df.groupby(['cluster4', 'Revenue']).size().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(6, 4))
count_revenue4.plot(kind='bar', stacked=True, ax=ax)
plt.title('Contagem de Revenue 0 e 1 por Cluster - 4 grupos')
st.pyplot(fig)

# Porcentagem de Revenue == True por Cluster
st.subheader("Porcentagem de Revenue 1 por Cluster - 3 e 4 grupos")
percentage_revenue_1_3 = df.groupby('cluster3')['Revenue'].mean() * 100

fig, ax = plt.subplots(figsize=(6, 4))
plt.bar(percentage_revenue_1_3.index, percentage_revenue_1_3.values)
plt.title('Porcentagem de Revenue 1 por Cluster - 3 grupos')
st.pyplot(fig)

percentage_revenue_1_4 = df.groupby('cluster4')['Revenue'].mean() * 100

fig, ax = plt.subplots(figsize=(6, 4))
plt.bar(percentage_revenue_1_4.index, percentage_revenue_1_4.values)
plt.title('Porcentagem de Revenue 1 por Cluster - 4 grupos')
st.pyplot(fig)

# Boxplot de BounceRates por Cluster
st.subheader("Distribuição de Bounce Rates por Cluster - 4 grupos")
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x='cluster4', y='BounceRates', data=df, ax=ax)
plt.title('Distribuição de Bounce Rates por Cluster')
st.pyplot(fig)

# Avaliação de Month por Cluster
st.subheader("Contagem de Month por Cluster")
count_month = df.groupby(['cluster4', 'Month']).size().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(10, 6))
count_month.plot(kind='bar', stacked=True, ax=ax)
plt.title('Contagem de Month por Cluster')
st.pyplot(fig)
