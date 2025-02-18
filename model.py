import numpy as np
import networkx as nx
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Các hàm kích hoạt
def tanh(x):
    return np.tanh(x)

# Hàm Graph Convolutional Layer
def graph_convolutional_layer(A, X, W, activation=None):
    N = A.shape[0]
    A_hat = A + np.eye(N)
    D_hat = np.sqrt(np.sum(A_hat, axis=0)) + 1e-5
    A_hat = A_hat / D_hat
    output = np.dot(A_hat, X)
    output = np.dot(output, W)
    
    if activation is not None:
        output = activation(output)

    return output

# Hàm GCN
def GCN(A, X, W1, W2, activation1=tanh, activation2=tanh):
    H1 = graph_convolutional_layer(A, X, W1, activation=activation1)
    H2 = graph_convolutional_layer(A, H1, W2, activation=activation2)
    return H2

# Xây dựng mô hình LSTM
def build_LSTM_model(vocab_size, embedding_dim, max_length):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
        LSTM(20, return_sequences=False),
        Dense(10)  
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Tạo đồ thị quan hệ giữa các câu
def build_sentence_relation_graph(sentences):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    similarity_matrix = cosine_similarity(tfidf_matrix)

    graph = nx.Graph()

    for i, sentence in enumerate(sentences):
        graph.add_node(i, text=sentence)

    num_sentences = len(sentences)
    for i in range(num_sentences):
        for j in range(i + 1, num_sentences):  
            weight = similarity_matrix[i, j]
            if weight > 0:
                graph.add_edge(i, j, weight=weight)

    return graph

# Chạy mô hình LSTM và GCN
def run_model(sentences, vocab_size, embedding_dim, max_length):
    # Bước 1: Encode câu với LSTM
    model_lstm = build_LSTM_model(vocab_size, embedding_dim, max_length)
    
    # Chuyển các câu thành ma trận số học (sử dụng LSTM)
    sequences = tokenizer.texts_to_sequences(sentences)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    
    sentence_embeddings = model_lstm.predict(padded_sequences)
    
    # Bước 2: Xây dựng đồ thị quan hệ giữa các câu
    graph = build_sentence_relation_graph(sentences)
    
    # Chuyển đồ thị quan hệ thành ma trận kề (adjacency matrix)
    A = nx.adjacency_matrix(graph).todense()
    A = np.array(A)
    
    # Bước 3: Áp dụng GCN
    N, D = sentence_embeddings.shape
    W1 = np.random.rand(D, 4)  # Hệ số học cho lớp 1
    W2 = np.random.rand(4, D)  # Hệ số học cho lớp 2
    
    final_embeddings = GCN(A, sentence_embeddings, W1, W2, activation1=tanh, activation2=tanh)
    
    return final_embeddings
