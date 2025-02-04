# Análise de Sentimento de Comentários  

**Um aplicativo de Machine Learning para analisar sentimentos de comentários em tempo real ou via Google Sheets!**  

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
- **Virtualenv** (opcional, mas recomendado)  

## ⚡ Funcionalidades  

# 🔍 Análise de Sentimentos em Tempo Real  
- Digite um comentário no campo de entrada e obtenha a previsão do sentimento (positivo, negativo ou neutro).  

# 📂 Processamento de Arquivos CSV  
- Faça upload de um arquivo CSV contendo comentários para análise em massa.  
- O modelo prevê os sentimentos para cada comentário e adiciona uma nova coluna com os resultados.  
- Opção de download do arquivo processado com os sentimentos classificados.  

# 📊 Visualização de Dados  
- Gráficos interativos para melhor compreensão da distribuição dos sentimentos.  
- Gráficos de barras empilhadas e lado a lado, utilizando **Plotly**.  
- Análises temporais para observar tendências ao longo do tempo.  

# 📗 Integração com Google Sheets  
- Analise sentimentos diretamente de uma planilha do **Google Sheets**.  
- Previsões são inseridas automaticamente na planilha original.  
- Interface intuitiva para inserir ID da planilha, aba e colunas de interesse.  


## Exemplos de uso

```python
# 🔍 Análise de Sentimento de um Comentário  
# Insira um comentário e analise seu sentimento  

text_input = "O produto é incrível e superou minhas expectativas!"

if text_input.strip():
    comments = pd.DataFrame([text_input], columns=["Comentario"])
    sentiment_pred_code = testar_comentarios_dataframe(comments, "Comentario", model, tokenizer)
    predicted_sentiment = mapear_sentimento(sentiment_pred_code, label_encoder)[0]

    print(f"Sentimento Previsto: {predicted_sentiment}")
