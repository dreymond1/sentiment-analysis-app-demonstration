import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.graph_objs import Sankey
import plotly.express as px
from collections import Counter
from wordcloud import WordCloud
import time
import re

# Rede neural
import numpy as np
from keras.models import load_model
import pickle
from keras.preprocessing.sequence import pad_sequences

# Autenticação do GSheets
import os
import requests
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# Configurando autenticação do Google Sheets
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
creds = None

# URL do arquivo token.json no GitHub
GITHUB_CREDENTIALS_URL = "https://github.com/dreymond1/streamlitapp/blob/main/token.json"

# Faz o download do arquivo token.json do GitHub
def download_credentials_from_github(url, filename="token.json"):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, "wb") as file:
            file.write(response.content)
    else:
        raise Exception(f"Erro ao baixar o arquivo: {response.status_code}")

# Baixar o credentials.json se não existir
if not os.path.exists('token.json'):
    download_credentials_from_github(GITHUB_CREDENTIALS_URL)

# Processo de autenticação
if os.path.exists('token.json'):
    creds = Credentials.from_authorized_user_file('token.json', SCOPES)

if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
        creds = flow.run_local_server(port=0)
    with open('token.json', 'w') as token:
        token.write(creds.to_json())

# Configurando o serviço do Google Sheets
service = build('sheets', 'v4', credentials=creds)
sheet = service.spreadsheets()
###

# Carregar o modelo, tokenizer e label_encoder (ajuste para o caminho do arquivo)
def carregar_modelo_e_tokenizer():
    model = load_model("files/sentiment_model.h5")  # Caminho para o modelo treinado
    with open("files/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("files/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    
    if not model or not tokenizer or not label_encoder:
        raise ValueError("Erro ao carregar modelo, tokenizer ou label_encoder. Verifique os arquivos!")
    
    return model, tokenizer, label_encoder



# Função para testar novos comentários (no caso, uma coluna de comentários de um DataFrame)
def testar_comentarios_dataframe(df, coluna_comentario, model, tokenizer, max_len_contexto=50):
    
    data_base = df[coluna_comentario].apply(substituir_termos)
    
    # Tokenização dos comentários da coluna
    X_novos_comentarios = tokenizer.texts_to_sequences(data_base.tolist())
    
    # Padding para garantir o mesmo tamanho
    X_novos_comentarios = pad_sequences(X_novos_comentarios, maxlen=max_len_contexto, padding='post')
    
    # Previsão
    predicoes = model.predict(X_novos_comentarios)
    
    # Decodificar as previsões
    y_pred = np.argmax(predicoes, axis=1)  # Pega a classe com maior probabilidade
    return y_pred

# Função para mapear a previsão de volta ao sentimento original
def mapear_sentimento(predicoes_codificadas, label_encoder):
    sentimentos_preditos = label_encoder.inverse_transform(predicoes_codificadas)
    return sentimentos_preditos
###

 # Função para aplicar as substituições
def substituir_termos(texto):
    for termo, substituto in substituicoes.items():
        texto = re.sub(termo, substituto, texto, flags=re.IGNORECASE)
    return texto

 # Dicionário de substituições
substituicoes = {
    r'\bnao\b': 'não',
    r'\bn\b': 'não',
    r'\brapido\b': 'rápido',
    r'\brapida\b': 'rápida',
    r'\brapidos\b': 'rápidos',
    r'\brapidas\b': 'rápidas',
    r'\bvoce\b': 'você',
    r'\bvoces\b': 'vocês',
    r'\bvc\b': 'você',
    r'\bvcs\b': 'vocês',
    r'\bcomentario\b': 'comentário',
    r'\bcomentarios\b': 'comentários',
    r'\bq\b': 'que',
    r'\bobg\b': 'obrigado',
    r'\bola\b': 'olá',
    r'\bfacil\b': 'fácil',
    r'\bdificil\b': 'difícil',
    r'\bdificeis\b': 'difíceis',
    r'\bagente\b': 'a gente',
    r'\bpratico\b': 'prático',
    r'\bpratica\b': 'prática',
    r'\bpq\b': 'por que',
    r'\bso\b': 'só',
    r'\bsao\b': 'são',
    r'\bagil\b': 'ágil',
    r'\bpessimo\b': 'péssimo',
    r'\bpessima\b': 'péssima',
    r'\bpessimos\b': 'péssimos',
    r'\bpessimas\b': 'péssimas',
    r'\bhorrivelll\b': 'horrível',
    r'\bhorivel\b': 'horrível',
    r'\bhorriveis\b': 'horríveis',
    r'\bhorrivel\b': 'horrível',
    r'\bagil\b': 'ágil',
    r'\bsimpatica\b': 'simpática',
    r'\binfelismente\b': 'infelizmente',
    r'\bproblematico\b': 'problemático',
    r'\bdecepçao\b': 'decepção',
    r'\binsuportavel\b': 'insuportável',
    r'\binsuportaveis\b': 'insuportáveis',
    r'\birreparavel\b': 'irreparável',
    r'\bindisponivel\b': 'indisponível',
    r'\bantipatico\b': 'antipático',
    r'\botimo\b': 'ótimo',
    r'\botimos\b': 'ótimos',
    r'\botima\b': 'ótima',
    r'\botimas\b': 'ótimas',
    r'\bageis\b': 'ágeis',
    r'\bfacil\b': 'fácil',
    r'\bfaceis\b': 'fáceis',
    r'\bincrivel\b': 'incrível',
    r'\bincriveis\b': 'incríveis',
    r'\bfantastico\b': 'fantástico',
    r'\bfantastica\b': 'fantástica',
    r'\bfantasticos\b': 'fantásticos',
    r'\bfantasticas\b': 'fantásticas',
    r'\bsatisfatorio\b': 'satisfatório',
    r'\bsatisfatoria\b': 'satisfatória',
    r'\bsatisfatorios\b': 'satisfatórios',
    r'\bsatisfatorias\b': 'satisfatórias',
    r'\bimpecavel\b': 'impecável',
    r'\bimpecaveis\b': 'impecáveis',
    r'\bvalera\b': 'valerá',
    r'\butil\b': 'útil',
    r'\binsatisfatoriob\b': 'insatisfatório',
    r'\binsatisfatoria\b': 'insatisfatória',
    r'\binsatisfatorios\b': 'insatisfatórios',
    r'\binsatisfatorias\b': 'insatisfatórias',
    r'\bdesconfortavel\b': 'desconfortável',
    r'\bdesconfortsveis\b': 'desconfortáveis',
    r'\bprejuizo\b': 'prejuízo',
    r'\bpreju\b': 'prejuízo',
    r'\bdificeis\b': 'difíceis',
    r'\bunico\b': 'único',
    r'\bunica\b': 'única',
    r'\bunicas\b': 'únicas',
    r'\bunicos\b': 'únicos',
    r'\bmais rapido\b': 'mais rápido',
    r'\bmais agil\b': 'mais ágil'
}

st.set_page_config(page_title="Análise de Sentimento",
                   page_icon="🔍",
                   layout="wide",
                   initial_sidebar_state="expanded"
                   )

# Barra lateral que analisa sentimentos de uma planilha sheets
with st.sidebar:

    st.header("Analisar sentimentos de uma planilha Sheets 📗")
    st.info("Os campos deverão ser preenchidos de forma correta. Caso contrário, a previsão não será bem sucedida.")
    # Inputs
    st.markdown("### Instruções estão dentro de cada campo")

    id_input = st.text_input("ID:", placeholder="Exemplo: 1YYvqp_w9zDIgjNHFC8mh7Rkku6gKRN7Rwo8ydHKCqVA")

    aba_input = st.text_input("Nome da Aba:", placeholder="Exemplo: Aba-teste")

    coluna_comentario_input = st.text_input("Nome da Coluna de Comentário:", placeholder="Exemplo: A")

    coluna_sentimento_input = st.text_input("Nome da Coluna de Sentimento:", placeholder="Exemplo: B")

    if id_input == "" or aba_input == "" or coluna_comentario_input == "" or coluna_sentimento_input == "":
        st.error("Preencha todos os campos antes de analisar!")
    else:
        if st.button("Analisar Sentimentos da Planilha Sheets"):
            # ID da planilha e nome da aba
            SPREADSHEET_ID = id_input
            SHEET_NAME = aba_input
            COMMENT_COLUMN = coluna_comentario_input
            SENTIMENT_COLUMN = coluna_sentimento_input
    
            # Carregar o modelo, tokenizer e label_encoder
            model, tokenizer, label_encoder = carregar_modelo_e_tokenizer()
    
            # Função que processa os comentários e analisa os sentimentos
            def process_comments_and_sentiments():
                # Obtendo os dados da planilha
                range_to_read = f"{SHEET_NAME}!{COMMENT_COLUMN}:{SENTIMENT_COLUMN}"
                result = service.spreadsheets().values().get(spreadsheetId=SPREADSHEET_ID, range=range_to_read).execute()
                rows = result.get('values', [])
    
                # Preparar os dados para escrita (incluindo cabeçalhos)
                updates = []
                for i, row in enumerate(rows):
                    if i == 0:
                        continue  # Ignorar cabeçalhos, se houver
                    comment = row[0] if len(row) > 0 else ""  # Verificar se há comentário
                    sentiment = row[1] if len(row) > 1 else ""  # Verificar se há sentimento
    
                    if comment:  # Só processa se houver comentário
                        # Substituir termos no comentário
                        comment = substituir_termos(comment)
    
                        if not sentiment:  # Só prevê sentimento se estiver vazio
                            # Prever sentimento utilizando o modelo Keras
                            sentiment_pred_code = testar_comentarios_dataframe(pd.DataFrame([comment], columns=["Comentario"]), "Comentario", model, tokenizer)
                            predicted_sentiment = mapear_sentimento(sentiment_pred_code, label_encoder)[0]
                            updates.append({'range': f"{SHEET_NAME}!{SENTIMENT_COLUMN}{i+1}",
                                            'values': [[predicted_sentiment]]})
    
                # Atualizar a planilha com os sentimentos
                if updates:
                    body = {'valueInputOption': 'RAW', 'data': updates}
                    service.spreadsheets().values().batchUpdate(spreadsheetId=SPREADSHEET_ID, body=body).execute()
    
                print(f"Processamento concluído!")
    
            # Executar
            process_comments_and_sentiments()



# Título e descrição do aplicativo
st.markdown("## 🔍 Análise de Sentimento de Comentários")
st.write(
    """
    Este aplicativo utiliza Machine Learning para prever o sentimento de um comentário.
    Basta inserir o texto e clicar em **Analisar Sentimento** para ver o resultado.
    """
)

# Entrada de Texto
st.markdown("### ✍️ Digite o comentário para análise:")


text_input = st.text_area(
    "Insira o comentário aqui:", 
    placeholder="Exemplo: O produto é incrível e superou minhas expectativas!"
)

#text_input = substituir_termos(text_input)

# Carregar modelo e tokenizer
model, tokenizer, label_encoder = carregar_modelo_e_tokenizer()

# Botão de Previsão

if st.button("Analisar Sentimento", key="btn1"):
    if text_input.strip():
        # Prever o sentimento do comentário
        comments = pd.DataFrame([text_input], columns=["Comentario"])
        sentiment_pred_code = testar_comentarios_dataframe(comments, "Comentario", model, tokenizer)
        predicted_sentiment = mapear_sentimento(sentiment_pred_code, label_encoder)[0]

        # Exibir resultado com formatação
        st.markdown("#### 🎯 Resultado da Análise:")
        if predicted_sentiment == "positivo": 
            st.success(f"Sentimento Previsto: **Positivo** 😊")
        elif predicted_sentiment == "negativo":
            st.error(f"Sentimento Previsto: **Negativo** 😠")
        else:
            st.info(f"Sentimento Previsto: **Neutro** 😐")
    else:
        st.warning("⚠️ Por favor, insira um comentário para analisar o sentimento.")

st.markdown("---")

# Upload de CSV para análise em massa
st.markdown("### 📂 Faça upload de um arquivo CSV com comentários:")
uploaded_file = st.file_uploader("Escolha um arquivo CSV (deve possuir pelo menos uma coluna chamada 'Comentário' e 'Motivo')", type=["csv"])

#sp3, col_indent2, sp4 = st.columns([0.00000000001, 5, 2])

  
if uploaded_file:
    # Carregar o arquivo CSV
    data = pd.read_csv(uploaded_file, encoding='iso-8859-1', sep=';', on_bad_lines='skip')
    
    # Apagar as linhas onde a coluna "Comentário" está vazia ou nula
    data = data.dropna(subset=['Comentário'])
    
    # Prever sentimentos para todos os comentários
    sentiment_codes = testar_comentarios_dataframe(data, 'Comentário', model, tokenizer)
    sentiments = mapear_sentimento(sentiment_codes, label_encoder)
    
    
    # Exibir mensagem e dados carregados
    st.write("📊 **Dados carregados com sucesso!**")
    st.dataframe(data.head())
    
    
    if 'Motivo' not in data.columns:
        st.warning("Aviso: A coluna 'Motivo' não foi encontrada no arquivo CSV.")
        
    else:
         # Criar a lista suspensa (droplist) com os itens únicos da coluna "Motivo" e ordenados alfabeticamente
        motivos_unicos = sorted(data['Motivo'].dropna().unique())
        motivo_selecionado = st.selectbox("Escolha um motivo", motivos_unicos)
    
        
        if st.button("Analisar sentimento específico"):                
            # Prever sentimentos para todos os comentários usando o modelo Keras e o tokenizer
            sentiment_codes = testar_comentarios_dataframe(data, 'Comentário', model, tokenizer)
            
            # Mapear as previsões codificadas para os sentimentos
            sentiments = mapear_sentimento(sentiment_codes, label_encoder)
        
            # Adicionar a coluna de sentimentos previstos ao DataFrame
            data['Sentimento'] = sentiments
    
            # Contagem de sentimentos para o motivo selecionado
            motivo_data = data[data['Motivo'] == motivo_selecionado]
            sentimento_count_motivo = motivo_data['Sentimento'].value_counts()
            
            # Exibir contagens
            st.markdown("##### 📊 Sentimentos identificados:")
            st.write(f"🟩 **Positivo:** {sentimento_count_motivo.get('positivo', 0)}")
            st.write(f"🟥 **Negativo:** {sentimento_count_motivo.get('negativo', 0)}")
            st.write(f"🟨 **Neutro:** {sentimento_count_motivo.get('neutro', 0)}")
        
            # GRAFICO POR MES        
            # Definir a ordem de empilhamento personalizada para as categorias "Sentimento"
            ordem_sentimentos = ['Positivo', 'Neutro', 'Negativo']
            
            # Passo 1: Tentar converter as datas de maneira flexível
            motivo_data['Data'] = pd.to_datetime(motivo_data['Data'], errors='coerce')
            
            # Passo 2: Tratar datas inválidas (caso você queira preencher com uma data padrão)
            # Por exemplo, vamos preencher as datas inválidas com a data mais próxima possível (ano de referência ou data fictícia)
            motivo_data['Data'] = motivo_data['Data'].fillna(method='bfill')
            
            # Passo 3: Alternativa 1 - Remover apenas as linhas com valores NaT na coluna 'Data' (se for essencial)
            motivo_data = motivo_data.dropna(subset=['Data'])
            
            # Passo 4: Alternativa 2 - Preencher os valores NaT com uma data específica (por exemplo, a data mínima ou uma data de referência)
            # Caso queira preencher as datas inválidas (não recomendado se não fizer sentido no seu contexto)
            # data['Data'] = data['Data'].fillna(pd.Timestamp('2000-01-01'))
            
            # Passo 5: Criar a coluna 'Mes_Ano' para agrupar por mês e ano
            motivo_data['Mes_Ano'] = motivo_data['Data'].dt.to_period('M')
            
            # Passo 6: Contar os sentimentos por mês
            grafico_sentimentos_mes = motivo_data.groupby(['Mes_Ano', 'Sentimento']).size().reset_index(name='Quantidade')
            
            # Passo 7: Ordenar os valores para visualização correta
            grafico_sentimentos_mes['Mes_Ano'] = grafico_sentimentos_mes['Mes_Ano'].astype(str)
            grafico_sentimentos_mes = grafico_sentimentos_mes.sort_values('Mes_Ano')
            
            # Passo 8: Criar o gráfico de barras empilhadas
            fig_sentimentos_mes = px.bar(
                grafico_sentimentos_mes, 
                x='Mes_Ano', 
                y='Quantidade', 
                color='Sentimento', 
                title='Sentimentos por Mês',
                labels={'Mes_Ano': 'Mês/Ano', 'Quantidade': 'Quantidade'},
                text='Quantidade',
                color_discrete_map={
                    'positivo': 'green',
                    'negativo': 'red',
                    'neutro': 'yellow'
                }
            )
            
            # Passo 9: Ajustar o layout do gráfico
            fig_sentimentos_mes.update_layout(
                xaxis_title='Mês/Ano',
                yaxis_title='Quantidade de Sentimentos',
                barmode='stack',
                xaxis=dict(type='category'),
                legend=dict(traceorder='reversed'), #Para reverter a ordem na legenda
                coloraxis_colorbar=dict(tickvals=[0, 1, 2], ticktext=ordem_sentimentos)
            )
            
            # Passo 10: Exibir o gráfico mês a mês
            st.plotly_chart(fig_sentimentos_mes)

        
    st.markdown("---")

    st.write("Selecione esse botão para analisar os sentimentos do arquivo importado:")
    # Botão para iniciar a análise em massa
    if st.button("Analisar Sentimentos no CSV"):
        if 'Comentário' not in data.columns:
            st.warning("Aviso: A coluna 'Comentário' não foi encontrada no arquivo CSV.")
        else:
            st.success("Arquivo CSV válido!")
            with st.spinner("Carregando..."):
                progress_bar = st.progress(0)
                for i in range(101):
                    time.sleep(0.05)
                    progress_bar.progress(i)
    
            # Prever sentimentos para todos os comentários usando o modelo Keras e o tokenizer
            sentiment_codes = testar_comentarios_dataframe(data, 'Comentário', model, tokenizer)
            
            # Mapear as previsões codificadas para os sentimentos
            sentiments = mapear_sentimento(sentiment_codes, label_encoder)
        
            # Adicionar a coluna de sentimentos previstos ao DataFrame
            data['Sentimento'] = sentiments
    
            # Exibir os dados com os sentimentos
            st.dataframe(data.head())
    
            # Gráficos de distribuição dos sentimentos usando plotly
            st.markdown("#### 📊 Visualização dos Sentimentos:")
    
            # Contagem de cada sentimento
            sentiment_count = data['Sentimento'].value_counts(normalize=True).reset_index()
            sentiment_count.columns = ['Sentimento', 'Proporcao']
    
            sentiment_count['Proporcao'] = sentiment_count['Proporcao'] * 100
    
            # Adicionar as porcentagens para cada sentimento
            sentiment_count['Proporcao_Label'] = sentiment_count['Proporcao'].round(2).astype(str) + '%'
    
            # Contagem de cada sentimento
            sentiment_count_2 = data['Sentimento'].value_counts()

            # Exibir contagens
            st.markdown("##### Sentimentos identificados:")
            st.write(f"🟩 **Positivo:** {sentiment_count_2.get('positivo', 0)}")
            st.write(f"🟥 **Negativo:** {sentiment_count_2.get('negativo', 0)}")
            st.write(f"🟨 **Neutro:** {sentiment_count_2.get('neutro', 0)}")

            col1, spacer, col2 = st.columns([7, 0.5, 7]) 
            
            
            # Criar gráfico de barras 100%
            fig_stack = go.Figure()
    
             # Barra para o Sentimento Negativo
            fig_stack.add_trace(go.Bar(
                x=['Sentimentos'],
                y=sentiment_count.loc[sentiment_count['Sentimento'] == 'negativo', 'Proporcao'],
                name='Negativo',
                marker=dict(color='red'),
                text=sentiment_count.loc[sentiment_count['Sentimento'] == 'negativo', 'Proporcao_Label'],
                hoverinfo='text',
            ))
    
            # Barra para o sentimento positivo
            fig_stack.add_trace(go.Bar(
                x=['Sentimentos'],
                y=sentiment_count.loc[sentiment_count['Sentimento'] == 'positivo', 'Proporcao'],
                name='Positivo',
                marker=dict(color='green'),
                text=sentiment_count.loc[sentiment_count['Sentimento'] == 'positivo', 'Proporcao_Label'],
                hoverinfo='text',
            ))
    
            # Barra para o sentimento seutro
            fig_stack.add_trace(go.Bar(
                x=['Sentimentos'],
                y=sentiment_count.loc[sentiment_count['Sentimento'] == 'neutro', 'Proporcao'],
                name='Neutro',
                marker=dict(color='yellow'),
                text=sentiment_count.loc[sentiment_count['Sentimento'] == 'neutro', 'Proporcao_Label'],
                hoverinfo='text',
            ))
    
    
            # Ajustar o layout
            fig_stack.update_layout(
                barmode='stack',
                title="Sentimentos empilhados",
                xaxis=dict(title=''),
                yaxis=dict(title='', range=[0, 100]),
                showlegend=True
            )
    
            #st.plotly_chart(fig_stack)
    
            # Criar gráfico de barras lado a lado
            fig_side_by_side = go.Figure()
    
            # Barra para o sentimento positivo
            fig_side_by_side.add_trace(go.Bar(
                x=['Sentimentos'],
                y=sentiment_count.loc[sentiment_count['Sentimento'] == 'positivo', 'Proporcao'],
                name='Positivo',
                marker=dict(color='green'),
                text=sentiment_count.loc[sentiment_count['Sentimento'] == 'positivo', 'Proporcao_Label'],
                hoverinfo='text',
            ))    
    
             # Barra para o sentimento neutro
            fig_side_by_side.add_trace(go.Bar(
                x=['Sentimentos'],
                y=sentiment_count.loc[sentiment_count['Sentimento'] == 'neutro', 'Proporcao'],
                name='Neutro',
                marker=dict(color='yellow'),
                text=sentiment_count.loc[sentiment_count['Sentimento'] == 'neutro', 'Proporcao_Label'],
                hoverinfo='text',
            ))
    
    
            # Barra para o sentimento negativo
            fig_side_by_side.add_trace(go.Bar(
                x=['Sentimentos'],
                y=sentiment_count.loc[sentiment_count['Sentimento'] == 'negativo', 'Proporcao'],
                name='Negativo',
                marker=dict(color='red'),
                text=sentiment_count.loc[sentiment_count['Sentimento'] == 'negativo', 'Proporcao_Label'],
                hoverinfo='text',
            ))
    
            # Ajustar o layout
            fig_side_by_side.update_layout(
                barmode='group',
                title="Sentimentos em colunas",
                xaxis=dict(title=''),
                yaxis=dict(title='', range=[0, 100]),
                showlegend=True
            )
    
            #st.plotly_chart(fig_side_by_side)

            with col1:
                st.plotly_chart(fig_stack, use_container_width=True) 
        
            with col2:
                st.plotly_chart(fig_side_by_side, use_container_width=True) 
                           
            if 'Data' not in data.columns:
                st.warning("Aviso: A coluna 'Data' não foi encontrada no arquivo CSV.")
            else:
                # GRAFICO POR MES        
                # Definir a ordem de empilhamento personalizada para as categorias "Sentimento"
                ordem_sentimentos = ['positivo', 'neutro', 'negativo']
                
                # Passo 1: Tentar converter as datas de maneira flexível
                data_mes = data
                data_mes['Data'] = pd.to_datetime(data['Data'], errors='coerce')
                
                # Passo 2: Tratar datas inválidas (caso você queira preencher com uma data padrão)
                # Por exemplo, vamos preencher as datas inválidas com a data mais próxima possível (ano de referência ou data fictícia)
                data_mes['Data'] = data_mes['Data'].fillna(method='bfill')
                
                # Passo 3: Alternativa 1 - Remover apenas as linhas com valores NaT na coluna 'Data' (se for essencial)
                data_mes = data_mes.dropna(subset=['Data'])
                
                # Passo 4: Alternativa 2 - Preencher os valores NaT com uma data específica (por exemplo, a data mínima ou uma data de referência)
                # Caso queira preencher as datas inválidas (não recomendado se não fizer sentido no seu contexto)
                # data['Data'] = data['Data'].fillna(pd.Timestamp('2000-01-01'))
                
                # Passo 5: Criar a coluna 'Mes_Ano' para agrupar por mês e ano
                data_mes['Mes_Ano'] = data_mes['Data'].dt.to_period('M')
                
                # Passo 6: Contar os sentimentos por mês
                grafico_sentimentos_mes = data_mes.groupby(['Mes_Ano', 'Sentimento']).size().reset_index(name='Quantidade')
                
                # Passo 7: Ordenar os valores para visualização correta
                grafico_sentimentos_mes['Mes_Ano'] = grafico_sentimentos_mes['Mes_Ano'].astype(str)
                grafico_sentimentos_mes = grafico_sentimentos_mes.sort_values('Mes_Ano')
                
                # Passo 8: Criar o gráfico de barras empilhadas
                fig_sentimentos_mes = px.bar(
                    grafico_sentimentos_mes, 
                    x='Mes_Ano', 
                    y='Quantidade', 
                    color='Sentimento', 
                    title='Sentimentos por Mês',
                    labels={'Mes_Ano': 'Mês/Ano', 'Quantidade': 'Quantidade'},
                    text='Quantidade',
                    color_discrete_map={
                        'positivo': 'green',
                        'negativo': 'red',
                        'neutro': 'yellow'
                    }
                )
                
                # Passo 9: Ajustar o layout do gráfico
                fig_sentimentos_mes.update_layout(
                    xaxis_title='Mês/Ano',
                    yaxis_title='Quantidade de Sentimentos',
                    barmode='stack',
                    xaxis=dict(type='category'),
                    legend=dict(traceorder='reversed'), #Para reverter a ordem na legenda
                    coloraxis_colorbar=dict(tickvals=[0, 1, 2], ticktext=ordem_sentimentos)
                )
                
                # Passo 10: Exibir o gráfico mês a mês
                st.plotly_chart(fig_sentimentos_mes)
            
            not_words = [
                'a', 'oi', '.', 'ola', ',', 'fazer', 'boa', 'está', 'tarde', 'teste', 'ok', 'olá', 'à', 'logo', 'desde', 'podem', 'além', 'q', 'sim', 'nao', 'falando', 'lá', 'meus', 'ficou', 'queren', 'sei', 'hoje', 'aqui', 'ficar', 'te', 'mas', 'neste', 'nesta', 'nessa', 'nesse', 'e', 'vou', 'vejo', 'entrará', 'estava', 'meu', 've', 'vê', 'ter', 'logo', 'fosse', 'horas', 'ainda', 'dia', 'falar', 'minuto', 'minutos', 'hora', 'pela', 'dar', 'então', 'sou', 'vou', 'ficaram', 'agora', 'os', 'me', 'algmas', 'algumas', 'alguns', 'ali', 'ambos', 'antes', 'ao', 'aos', 'apenas', 'apoio', 'apos', 'após', 'aquela', 'aquelas', 'aquele', 'aqueles', 'aquilo', 'as', 'às', 'ate', 'até', 'atras', 'atrás', 'bem', 'bom', 'cada', 'certa', 'certas', 'certeza', 'certo', 'certos', 'com', 'como', 'conforme', 'contra', 'contudo', 'da', 'da', 'dá', 'dado', 'das', 'de', 'dela', 'delas', 'dele', 'deles', 'dessa', 'dessas', 'desse', 'desses', 'desta', 'destas', 'deste', 'destes', 'deve', 'devem', 'devera', 'deverá', 'deverao', 'deverão', 'deveria', 'deveriam', 'disse', 'diz', 'dizem', 'dizer', 'do', 'dos', 'duas', 'duplo', 'duplos', 'ela', 'elas', 'ele', 'eles', 'em', 'em', 'enquanto', 'essa', 'essas', 'esse', 'esses', 'esta', 'estamos', 'estão', 'este', 'estes', 'essa', 'esses', 'e', 'eu', 'ela', 'elas', 'isto', 'isso', 'isso', 'isto', 'ja', 'já', 'jamais', 'jamas', 'lugar', 'mais', 'mas', 'mesmo', 'mesmos', 'muito', 'muitos', 'na', 'nas', 'no', 'nos', 'não', 'nós', 'nem', 'nosso', 'nossos', 'ou', 'outra', 'outras', 'outro', 'outros', 'para', 'para', 'para', 'pelo', 'pelas', 'pelo', 'perante', 'pois', 'por', 'porque', 'portanto', 'posso', 'pouca', 'poucas', 'pouco', 'poucos', 'primeiro', 'propria', 'própria', 'próprias', 'próprio', 'próprios', 'quais', 'qual', 'qualquer', 'quando', 'quanto', 'quantos', 'que', 'quem', 'quer', 'quero', 'se', 'seja', 'sejam', 'sejamos', 'sem', 'sempre', 'sendo', 'ser', 'sera', 'será', 'serao', 'serão', 'seria', 'seriam', 'seu', 'seus', 'si', 'sido', 'so', 'só', 'sob', 'sobre', 'sua', 'suas', 'talvez', 'tambem', 'também', 'tanta', 'tantas', 'tanto', 'tao', 'tão', 'te', 'tem', 'temos', 'tendo', 'tenha', 'tenham', 'tenhamos', 'tenho', 'tens', 'ter', 'tera', 'terá', 'terao', 'terão', 'teria', 'teriam', 'teu', 'teus', 'teve', 'tinha', 'tinham', 'tive', 'tivemos', 'tiver', 'tivera', 'tiveram', 'tiverem', 'tivermos', 'tivesse', 'tivessem', 'tivessemos', 'tivéssemos', 'toda', 'todas', 'todo', 'todos', 'tu', 'tua', 'tuas', 'tudo', 'ultimo', 'último', 'um', 'uma', 'umas', 'uns', 'vai', 'vao', 'vão', 'vem', 'vêm', 'vendo', 'ver', 'vez', 'vindo', 'vir', 'voce', 'você', 'voces', 'vocês', 'vos'
                'a', 'o', 'é', 'mim', 'pra', 'há', 'foi', 'à', 'ainda', 'agora', 'fui', 'estou', 'depois', 'meu', 'p', '','algmas', 'algumas', 'alguns', 'ali', 'ambos', 'antes', 'ao', 'aos', 'apenas', 'apoio', 'apos', 'após', 'aquela', 'aquelas', 'aquele', 'aqueles', 'aquilo', 'as', 'às', 'ate', 'até', 'atras', 'atrás', 'bem', 'bom', 'cada', 'certa', 'certas', 'certeza', 'certo', 'certos', 'com', 'como', 'conforme', 'contra', 'contudo', 'da', 'da', 'dá', 'dado', 'das', 'de', 'dela', 'delas', 'dele', 'deles', 'dessa', 'dessas', 'desse', 'desses', 'desta', 'destas', 'deste', 'destes', 'deve', 'devem', 'devera', 'deverá', 'deverao', 'deverão', 'deveria', 'deveriam', 'disse', 'diz', 'dizem', 'dizer', 'do', 'dos', 'duas', 'duplo', 'duplos', 'ela', 'elas', 'ele', 'eles', 'em', 'em', 'enquanto', 'essa', 'essas', 'esse', 'esses', 'esta', 'estamos', 'estão', 'este', 'estes', 'essa', 'esses', 'e', 'eu', 'ela', 'elas', 'isto', 'isso', 'isso', 'isto', 'ja', 'já', 'jamais', 'jamas', 'lugar', 'mais', 'mas', 'mesmo', 'mesmos', 'muito', 'muitos', 'na', 'nas', 'no', 'nos', 'não', 'nós', 'nem', 'nosso', 'nossos', 'ou', 'outra', 'outras', 'outro', 'outros', 'para', 'para', 'para', 'pelo', 'pelas', 'pelo', 'perante', 'pois', 'por', 'porque', 'portanto', 'posso', 'pouca', 'poucas', 'pouco', 'poucos', 'primeiro', 'propria', 'própria', 'próprias', 'próprio', 'próprios', 'quais', 'qual', 'qualquer', 'quando', 'quanto', 'quantos', 'que', 'quem', 'quer', 'quero', 'se', 'seja', 'sejam', 'sejamos', 'sem', 'sempre', 'sendo', 'ser', 'sera', 'será', 'serao', 'serão', 'seria', 'seriam', 'seu', 'seus', 'si', 'sido', 'so', 'só', 'sob', 'sobre', 'sua', 'suas', 'talvez', 'tambem', 'também', 'tanta', 'tantas', 'tanto', 'tao', 'tão', 'te', 'tem', 'temos', 'tendo', 'tenha', 'tenham', 'tenhamos', 'tenho', 'tens', 'ter', 'tera', 'terá', 'terao', 'terão', 'teria', 'teriam', 'teu', 'teus', 'teve', 'tinha', 'tinham', 'tive', 'tivemos', 'tiver', 'tivera', 'tiveram', 'tiverem', 'tivermos', 'tivesse', 'tivessem', 'tivessemos', 'tivéssemos', 'toda', 'todas', 'todo', 'todos', 'tu', 'tua', 'tuas', 'tudo', 'ultimo', 'último', 'um', 'uma', 'umas', 'uns', 'vai', 'vao', 'vão', 'vem', 'vêm', 'vendo', 'ver', 'vez', 'vindo', 'vir', 'voce', 'você', 'voces', 'vocês', 'vos'
            ]
            
             # Criar diagrama de Sankey
            st.markdown("#### 🧠 Diagrama de Sankey:")
    
            palavras_positivas = data[data['Sentimento'] == 'positivo']['Comentário'].dropna().astype(str).str.lower()
            palavras_negativas = data[data['Sentimento'] == 'negativo']['Comentário'].dropna().astype(str).str.lower()
            palavras_neutras = data[data['Sentimento'] == 'neutro']['Comentário'].dropna().astype(str).str.lower()
    
            # Remover palavras irrelevantes
            palavras_positivas = ' '.join(palavras_positivas.apply(lambda x: ' '.join([word for word in x.split() if word not in not_words])))
            palavras_negativas = ' '.join(palavras_negativas.apply(lambda x: ' '.join([word for word in x.split() if word not in not_words])))
            palavras_neutras = ' '.join(palavras_neutras.apply(lambda x: ' '.join([word for word in x.split() if word not in not_words])))
    
            prop_pos = int((sentiment_count_2.get('positivo', 0)/sentiment_count_2.sum()).round(1)*10)+2
            prop_neg = int((sentiment_count_2.get('negativo', 0)/sentiment_count_2.sum()).round(1)*10)+2
            prop_neu = int((sentiment_count_2.get('neutro', 0)/sentiment_count_2.sum()).round(1)*10)+2
    
            # Pegando as palavras de forma proporcional
            freq_positivas = Counter(palavras_positivas.split()).most_common(prop_pos)
            freq_negativas = Counter(palavras_negativas.split()).most_common(prop_neg)
            freq_neutras = Counter(palavras_neutras.split()).most_common(prop_neu)
    
            sentiment_count_2.get('Neutro', 0)
    
            # Criando as ligações para o gráfico de Sankey
            sentimentos = ['positivo', 'negativo', 'neutro']
    
            ligacoes = []
    
            for sentimento, palavras_frequentes in zip(sentimentos, [freq_positivas, freq_negativas, freq_neutras]):
                for palavra, _ in palavras_frequentes:
                    ligacoes.append((sentimento, palavra))
    
            palavras = [ligacao[1] for ligacao in ligacoes]
            sentimentos = [ligacao[0] for ligacao in ligacoes]
    
            # Criar o gráfico de Sankey
            # Gerar os rótulos para os sentimentos e palavras
            labels = list(set(palavras + sentimentos))
            label_to_index = {label: idx for idx, label in enumerate(labels)}
    
            # Construir as fontes e destinos
            origem = [label_to_index[sentimento] for sentimento in sentimentos]
            destino = [label_to_index[palavra] for palavra in palavras]
    
            # Valores das ligações (contagem de ocorrências)
            valores = [1] * len(origem)
    
            # Definindo a cor de cada sentimento
            link_colors = ['#a9f0a1' if sentimento == 'positivo' else '#f57171' if sentimento == 'negativo' else '#f7f55c' 
                   for sentimento in sentimentos]
    
            node_border = ['#a9f0a1' if sentimento == 'positivo' else '#f57171' if sentimento == 'negativo' else '#f7f55c' 
                   for sentimento in sentimentos]
    
            # Criar gráfico de Sankey
            fig_sankey = go.Figure(go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color=node_border, width=0.5),
                    color="gray",
                    label=labels
                ),
                link=dict(
                    source=origem,
                    target=destino,
                    value=valores,
                    color=link_colors
                )
            ))
    
            fig_sankey.update_layout(font=dict(color='#FFFFFF', size=13))
            st.plotly_chart(fig_sankey)
    
            # Gerar uma WordCloud para cada sentimento
            st.markdown("#### ☁️ Nuvens de Palavras por Sentimento:")
    
            # Filtrar comentários por sentimento
            comments_positive = data[data['Sentimento'] == 'positivo']['Comentário'].dropna().astype(str).str.lower()
            comments_negative = data[data['Sentimento'] == 'negativo']['Comentário'].dropna().astype(str).str.lower()
            comments_neutral = data[data['Sentimento'] == 'neutro']['Comentário'].dropna().astype(str).str.lower()
    
            # Gerar uma única string para cada sentimento
            positive_words = ' '.join(comments_positive)
            negative_words = ' '.join(comments_negative)
            neutral_words = ' '.join(comments_neutral)
    
            # Remover palavras irrelevantes
            positive_filtered = ' '.join(word for word in positive_words.split() if word not in not_words)
            negative_filtered = ' '.join(word for word in negative_words.split() if word not in not_words)
            neutral_filtered = ' '.join(word for word in neutral_words.split() if word not in not_words)
    
            # Criar as nuvens de palavras para cada sentimento
            positive_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_filtered)
            negative_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(negative_filtered)
            neutral_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(neutral_filtered)

            n1, spacer, n2, spacer, n3 = st.columns([7, 0.5, 7, 0.5, 7]) 
                
            # Exibir as nuvens de palavras para cada sentimento usando plotly
            
            with n1:
                st.markdown("**🟩 Positivo:**")
                fig_positive = go.Figure(go.Image(z=positive_wordcloud.to_array()))
                fig_positive.update_layout(
                    xaxis=dict(visible=False),  # Oculta o eixo X
                    yaxis=dict(visible=False),  # Oculta o eixo Y
                    margin=dict(l=0, r=0, t=0, b=0)  # Remove as margens
                )
                st.plotly_chart(fig_positive, use_container_width=True)
    
            
            with n2:
                st.markdown("**🟥 Negativo:**")
                fig_negative = go.Figure(go.Image(z=negative_wordcloud.to_array()))
                fig_negative.update_layout(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    margin=dict(l=0, r=0, t=0, b=0)
                )
                st.plotly_chart(fig_negative, use_container_width=True)
    
            
            with n3:
                st.markdown("**🟨 Neutro:**")
                fig_neutral = go.Figure(go.Image(z=neutral_wordcloud.to_array()))
                fig_neutral.update_layout(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    margin=dict(l=0, r=0, t=0, b=0)
                )
                st.plotly_chart(fig_neutral, use_container_width=True)
    
            st.success("Tudo pronto!")
    
            # Download dos resultados em CSV
            st.markdown("#### 📥 Baixe os resultados:")
            csv_result = data.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV com Sentimentos",
                data=csv_result,
                file_name="resultado_sentimentos.csv",
                mime="text/csv"
            )

# Rodapé
st.markdown("---")
st.markdown("**Criado por [Andrey Alves](https://github.com/dreymond1)** 🚀")
