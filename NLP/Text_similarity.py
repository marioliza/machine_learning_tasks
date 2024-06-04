import streamlit as st
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd


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
#st.image('./bivien.png',caption='Bivien',use_column_width='never', width = 60)
st.title("Aplicacion similitud de texto - PDV")
st.write("Realiza la carga del listado de puntos de venta latam a homologar y el listado de puntos de venta de la compañia. Debemos de garantizar que la primera colomuna tenga los nombres de puntos de venta a homologar, para los dos archivos y estos deben de ser en formato csv")

# File upload
uploaded_file1 = st.file_uploader("Carga el archivo de Latam", type="csv")
uploaded_file2 = st.file_uploader("Carga el archivo de pdv", type="csv")

if uploaded_file1 is not None and uploaded_file2 is not None:
    # Read the CSV files into DataFrames
    df1 = pd.read_csv(uploaded_file1)
    df2 = pd.read_csv(uploaded_file2)

    # Assuming the words are in the first column of each DataFrame
    words_db1 = df1.iloc[:, 0].tolist()
    words_db2 = df2.iloc[:, 0].tolist()

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
    model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')

    # Get embeddings for both lists of words
    embeddings_db1 = get_word_embeddings(words_db1, tokenizer, model)
    embeddings_db2 = get_word_embeddings(words_db2, tokenizer, model)

    # Compute similarities
    similarities = compute_similarity(embeddings_db1, embeddings_db2)

    # Find the top 3 most similar words for each word in words_db1 and create a DataFrame
    top_k = 10
    data = []

    for i, word in enumerate(words_db1):
        sim_scores = similarities[i]
        top_k_indices = torch.topk(sim_scores, top_k).indices
        for idx in top_k_indices:
            data.append({
                'Word_from_db1': word,
                'Best_Match_in_db2': words_db2[idx],
                'Similarity_Score': sim_scores[idx].item()
            })

    result_df = pd.DataFrame(data)

    # Display the DataFrame in the Streamlit app
    st.write("Top 3 matches for each word:")
    st.dataframe(result_df)

    # Provide option to download the DataFrame as an Excel file
    output_file = 'top_word_matches.xlsx'
    result_df.to_excel(output_file, index=False)
    st.download_button(
        label="Download as Excel",
        data=open(output_file, 'rb').read(),
        file_name=output_file,
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
