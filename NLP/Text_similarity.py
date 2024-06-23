import streamlit as st
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
from fuzzywuzzy import fuzz


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Function to get embeddings for a list of words
def get_word_embeddings(words, tokenizer, model):
    encoded_input = tokenizer(words, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    word_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return word_embeddings


# Function to compute cosine similarity between all pairs of word embeddings
def compute_similarity(embeddings1, embeddings2):
    similarities = torch.mm(embeddings1, embeddings2.T)  # Dot product
    norm1 = torch.norm(embeddings1, dim=1, keepdim=True)
    norm2 = torch.norm(embeddings2, dim=1, keepdim=True)
    similarities = similarities / (norm1 * norm2.T)  # Normalize to get cosine similarity
    return similarities


# Streamlit app
st.title("Aplicación de Similitud de Texto - PDV")
st.write(
    "Carga los listados de puntos de venta LATAM y de la compañía en formato CSV. La primera columna debe contener los nombres de los puntos de venta a homologar.")

# File upload
uploaded_file1 = st.file_uploader("Carga el archivo de LATAM", type="csv")
uploaded_file2 = st.file_uploader("Carga el archivo de PDV", type="csv")

if uploaded_file1 is not None and uploaded_file2 is not None:
    # Read the CSV files into DataFrames
    df1 = pd.read_csv(uploaded_file1, encoding='latin1')
    df2 = pd.read_csv(uploaded_file2, encoding='latin1')

    # Assuming the words are in the first column of each DataFrame
    words_db1 = df1.iloc[:, 0].tolist()
    words_db2 = df2.iloc[:, 0].tolist()

    # Load the transformer model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

    # Get embeddings for both lists of words using transformers
    embeddings_db1 = get_word_embeddings(words_db1, tokenizer, model)
    embeddings_db2 = get_word_embeddings(words_db2, tokenizer, model)

    # Compute similarities using transformers
    similarities = compute_similarity(embeddings_db1, embeddings_db2)

    # Find the top 5 most similar words for each word in words_db1 using transformers
    top_k = 5
    data_transformers = []

    for i, word in enumerate(words_db1):
        sim_scores = similarities[i]
        top_k_indices = torch.topk(sim_scores, top_k).indices
        for idx in top_k_indices:
            data_transformers.append({
                'Word_from_db1': word,
                'Best_Match_in_db2': words_db2[idx],
                'Similarity_Score': sim_scores[idx].item(),
                'Modelo': 'Transformers'
            })

    df_transformers = pd.DataFrame(data_transformers)

    # Compute similarities using FuzzyWuzzy
    data_fuzzy = []

    for word1 in words_db1:
        similarities_fuzzy = []
        for word2 in words_db2:
            similarity = fuzz.ratio(word1, word2)
            similarities_fuzzy.append((word2, similarity))

        similarities_fuzzy = sorted(similarities_fuzzy, key=lambda x: x[1], reverse=True)[:top_k]

        for word2, similarity in similarities_fuzzy:
            data_fuzzy.append({
                'Word_from_db1': word1,
                'Best_Match_in_db2': word2,
                'Similarity_Score': similarity,
                'Modelo': 'FuzzyWuzzy'
            })

    df_fuzzy = pd.DataFrame(data_fuzzy)

    # Load the SentenceTransformer model
    model_sentence_transformer = SentenceTransformer('thuan9889/llama_embedding_model_v1')

    # Get embeddings for both lists of words using SentenceTransformer
    embeddings_db1_st = model_sentence_transformer.encode(words_db1)
    embeddings_db2_st = model_sentence_transformer.encode(words_db2)

    # Compute similarities using SentenceTransformer
    embeddings_db1_st = torch.tensor(embeddings_db1_st)
    embeddings_db2_st = torch.tensor(embeddings_db2_st)
    similarities_st = compute_similarity(embeddings_db1_st, embeddings_db2_st)

    # Find the top 5 most similar words for each word in words_db1 using SentenceTransformer
    data_st = []

    for i, word in enumerate(words_db1):
        sim_scores = similarities_st[i]
        top_k_indices = torch.topk(sim_scores, top_k).indices
        for idx in top_k_indices:
            data_st.append({
                'Word_from_db1': word,
                'Best_Match_in_db2': words_db2[idx],
                'Similarity_Score': sim_scores[idx].item(),
                'Modelo': 'SentenceTransformer'
            })

    df_st = pd.DataFrame(data_st)

    # Concatenate the results
    df_result = pd.concat([df_transformers, df_fuzzy, df_st], ignore_index=True)

    # Display the DataFrame in the Streamlit app
    st.write("Top 5 matches for each word:")
    st.dataframe(df_result)

    # Provide option to download the DataFrame as an Excel file
    output_file = 'top_word_matches.xlsx'
    df_result.to_excel(output_file, index=False)
    st.download_button(
        label="Download as Excel",
        data=open(output_file, 'rb').read(),
        file_name=output_file,
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
