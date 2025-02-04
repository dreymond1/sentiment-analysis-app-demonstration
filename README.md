# Análise de Sentimento de Comentários  

**Um aplicativo de Machine Learning para analisar sentimentos de comentários em tempo real ou via Google Sheets!**  

### Objetivo
Dar uma maior autonomia aos stackeholders com a criação de ferramentas low-code/no-code, promovendo a democratização de análises e levantamento de insights

#### **Quais os ganhos?**

**Autonomia e Agilidade**: Ferramentas LCNC permitem que stakeholders criem e personalizem soluções, acelerando processos de decisão e diminuindo dependências de TI.

**Democratização e Acesso a Dados**: A análise de dados e insights se tornam acessíveis a todos, promovendo uma tomada de decisão mais informada e rápida em diversos níveis da organização.

**Redução de Custos e Inovação**: Menor necessidade de desenvolvedores, com soluções escaláveis e inovadoras sendo criadas diretamente pelas equipes de negócios.

## 📌 Sobre o Projeto  

Este projeto é uma aplicação **Streamlit** que utiliza um modelo de **Machine Learning** para prever o sentimento de textos, sendo capaz de:  

✅ **Analisar sentimentos** (positivo, negativo ou neutro) de comentários individuais.  
✅ **Processar grandes volumes de dados** importando planilhas do **Google Sheets** ou arquivos **CSV**.  
✅ **Visualizar insights** através de gráficos interativos com **Plotly**.  

A aplicação pode ser utilizada para monitorar feedbacks de clientes, analisar avaliações de produtos ou qualquer outro conjunto de comentários.  

---

## 🛠️ Tecnologias Utilizadas  

- **Python** (Streamlit, Pandas, NumPy, Keras, Plotly, WordCloud)  
- **Machine Learning** (Modelo de rede neural para análise de sentimentos)  
- **Google Sheets API** (Integração para análise de sentimentos diretamente em planilhas)  
- **Autenticação OAuth** para acesso seguro ao Google Sheets  

---

## Como Rodar o Projeto  

### 📌 Pré-requisitos  

Antes de começar, você precisará ter os seguintes itens instalados:  

- **Python 3.8+**  
- **pip** (gerenciador de pacotes do Python) 

## ⚡ Funcionalidades  

### Análise de Sentimentos em Tempo Real  
- Digite um comentário no campo de entrada e obtenha a previsão do sentimento (positivo, negativo ou neutro).  

### Processamento de Arquivos CSV  
- Faça upload de um arquivo CSV contendo comentários para análise em massa.  
- O modelo prevê os sentimentos para cada comentário e adiciona uma nova coluna com os resultados.  
- Opção de download do arquivo processado com os sentimentos classificados.  

### Visualização de Dados  
- Gráficos interativos para melhor compreensão da distribuição dos sentimentos.  
- Gráficos de barras empilhadas e lado a lado, utilizando **Plotly**.  
- Análises temporais para observar tendências ao longo do tempo.  

### Integração com Google Sheets  
- Analise sentimentos diretamente de uma planilha do **Google Sheets**.  
- Previsões são inseridas automaticamente na planilha original.  
- Interface intuitiva para inserir ID da planilha, aba e colunas de interesse.  


## Exemplos de uso

### 1. Análise individual
```python
# 🔍 Análise de Sentimento de um Comentário  
# Insira um comentário e analise seu sentimento  

text_input = "O produto é incrível e superou minhas expectativas!"

if text_input.strip():
    comments = pd.DataFrame([text_input], columns=["Comentario"])
    sentiment_pred_code = testar_comentarios_dataframe(comments, "Comentario", model, tokenizer)
    predicted_sentiment = mapear_sentimento(sentiment_pred_code, label_encoder)[0]

    print(f"Sentimento Previsto: {predicted_sentiment}")

```
### 2. Análise em massa

```python
# 📂 Analisando Sentimentos em um Arquivo CSV  
# Faça upload de um arquivo CSV contendo comentários e processe a análise em massa  

uploaded_file = "comentarios.csv"  # Nome do arquivo CSV

if uploaded_file:
    data = pd.read_csv(uploaded_file, encoding='iso-8859-1', sep=';', on_bad_lines='skip')
    data = data.dropna(subset=['Comentário'])  # Remover linhas sem comentário

    sentiment_codes = testar_comentarios_dataframe(data, 'Comentário', model, tokenizer)
    sentiments = mapear_sentimento(sentiment_codes, label_encoder)
    data['Sentimento'] = sentiments  # Adicionar a coluna de sentimentos

    print(data.head())  # Exibir os primeiros resultados
```
### 3. Análise dentro de um material sheets

```python
# 📗 Analisando Sentimentos de uma Planilha Google Sheets  
# Preencha os campos com o ID da planilha, nome da aba e colunas  

SPREADSHEET_ID = "1YYvqp_w9zDIgjNHFC8mh7Rkku6gKRN7Rwo8ydHKCqVA"
SHEET_NAME = "Aba-teste"
COMMENT_COLUMN = "A"
SENTIMENT_COLUMN = "B"

model, tokenizer, label_encoder = carregar_modelo_e_tokenizer()

def process_comments_and_sentiments():
    range_to_read = f"{SHEET_NAME}!{COMMENT_COLUMN}:{SENTIMENT_COLUMN}"
    result = service.spreadsheets().values().get(spreadsheetId=SPREADSHEET_ID, range=range_to_read).execute()
    rows = result.get('values', [])

    updates = []
    for i, row in enumerate(rows):
        if i == 0:
            continue
        comment = row[0] if len(row) > 0 else ""
        sentiment = row[1] if len(row) > 1 else ""

        if comment:
            comment = substituir_termos(comment)
            if not sentiment:
                sentiment_pred_code = testar_comentarios_dataframe(pd.DataFrame([comment], columns=["Comentario"]), "Comentario", model, tokenizer)
                predicted_sentiment = mapear_sentimento(sentiment_pred_code, label_encoder

```

## 👨‍💻 Autor  

Feito por **Andrey Alves** (https://www.linkedin.com/in/andrey-de-abreu-9a499b154/)  
Analista de Dados e entusiasta em ciência e engenharia de dados  
Desenvolvedor de soluções inovadoras para análise de dados e automação.
