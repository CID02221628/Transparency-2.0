import os
import json
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import pipeline as transformers_pipeline
from bertopic import BERTopic
from gensim import corpora, models
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px
import plotly.graph_objects as go
import torch
from collections import Counter
import io
import base64
from wordcloud import WordCloud
from datasets import load_dataset
from concurrent.futures import ProcessPoolExecutor, as_completed 
import psutil

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Load configuration
current_directory = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_directory, 'config.json')

# Load configuration with error handling
try:
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    print(f"Loaded configuration from {config_path}")
except FileNotFoundError:
    print(f"Configuration file not found at {config_path}")
    config = {}  # Default to empty config
except json.JSONDecodeError as e:
    print(f"Error decoding JSON from {config_path}: {e}")
    config = {}

from datasets import load_dataset

class TextProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        print("Initialized TextProcessor")

    def load_texts_from_dataset(self, dataset_name='ag_news', split='train', sample_size=None):
        """Load texts from a specified dataset using the datasets library."""
        try:
            dataset = load_dataset(dataset_name, split=split)
            print(f"Loaded {len(dataset)} records from the {dataset_name} dataset.")
            
            # Shuffle and sample the dataset randomly each time
            if sample_size and sample_size < len(dataset):
                dataset = dataset.shuffle(seed=None).select(range(sample_size))
                print(f"Randomly sampled {sample_size} records from the dataset.")

            texts = [record['text'] for record in dataset]
            labels = [record['label'] for record in dataset]  # Labels can be used if needed
            # Generate pseudo file paths for compatibility
            file_paths = [f"{dataset_name}_{split}_article_{i}.txt" for i in range(len(texts))]
            return texts, file_paths
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            return [], []

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        print("Initialized TextProcessor")

    def load_texts_from_dataset(self, dataset_name='ag_news', split='train', sample_size=None):
        """Load texts from a specified dataset using the datasets library."""
        try:
            dataset = load_dataset(dataset_name, split=split)
            print(f"Loaded {len(dataset)} records from the {dataset_name} dataset.")
            
            # Shuffle the dataset before sampling
            dataset = dataset.shuffle(seed=42)
            
            # If sample_size is specified and less than dataset size, sample the dataset
            if sample_size and sample_size < len(dataset):
                dataset = dataset.select(range(sample_size))
                print(f"Randomly sampled {sample_size} records from the dataset.")

            texts = [record['text'] for record in dataset]
            labels = [record['label'] for record in dataset]  # Labels can be used if needed
            # Generate pseudo file paths for compatibility
            file_paths = [f"{dataset_name}_{split}_article_{i}.txt" for i in range(len(texts))]
            return texts, file_paths
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            return [], []

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        print("Initialized TextProcessor")

    def load_texts_from_folder(self, folder_path):
        """Load all text files from a given folder."""
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} does not exist. Exiting.")
            return [], []
        
        texts, file_paths = [], []
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path) and file_path.endswith('.txt'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        texts.append(file.read())
                        file_paths.append(file_path)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
        print(f"Loaded {len(texts)} text files from {folder_path}")
        return texts, file_paths

    def load_texts_from_dataset(self, dataset_name='ag_news', split='train', sample_size=None):
        """Load texts from a specified dataset using the datasets library."""
        try:
            dataset = load_dataset(dataset_name, split=split)
            print(f"Loaded {len(dataset)} records from the {dataset_name} dataset.")
            
            # If sample_size is specified and less than dataset size, sample the dataset
            if sample_size and sample_size < len(dataset):
                dataset = dataset.shuffle(seed=42).select(range(sample_size))
                print(f"Sampled {sample_size} records from the dataset.")

            texts = [record['text'] for record in dataset]
            labels = [record['label'] for record in dataset]  # Labels can be used if needed
            # Generate pseudo file paths for compatibility
            file_paths = [f"{dataset_name}_{split}_article_{i}.txt" for i in range(len(texts))]
            return texts, file_paths
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            return [], []

    def tokenize_text(self, text):
        return word_tokenize(text.lower())

    def filter_tokens(self, tokens):
        return [word for word in tokens if word.isalpha() and word not in self.stop_words]

    def preprocess_texts(self, texts):
        tokenized_texts = [self.tokenize_text(text) for text in texts]
        processed_texts = [self.filter_tokens(tokens) for tokens in tokenized_texts]
        return processed_texts

class SentimentAnalyzer:
    def __init__(self, model_name=config.get('sentiment_model', 'distilbert-base-uncased'), temperature=1.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sentiment_pipeline = transformers_pipeline("sentiment-analysis", model=model_name, device=0 if torch.cuda.is_available() else -1)
        self.temperature = temperature
        print("Initialized SentimentAnalyzer with model:", model_name)

    def sliding_window_chunks(self, text, max_length=config.get('max_length', 512), stride=config.get('stride', 128)):
        tokens = self.sentiment_pipeline.tokenizer(text, truncation=True, max_length=max_length, return_tensors='pt')['input_ids'][0]
        total_tokens = tokens.size(0)
        chunks = []
        for i in range(0, total_tokens, stride):
            chunk_tokens = tokens[i:i + max_length]
            chunk_text = self.sentiment_pipeline.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)
            if i + max_length >= total_tokens:
                break
        return chunks

    def softmax_with_temperature(self, logits):
        logits = logits / self.temperature
        return torch.nn.functional.softmax(logits, dim=-1)

    def sentiment_analysis_chunk(self, text, file_path):
        chunks = self.sliding_window_chunks(text)
        chunk_sentiments = []
        for chunk in chunks:
            logits = self.sentiment_pipeline(chunk, return_all_scores=True)[0]
            scores = self.softmax_with_temperature(torch.tensor([score['score'] for score in logits]))
            chunk_sentiments.append(scores[1].item())
        return chunk_sentiments, [file_path] * len(chunk_sentiments), [f"Chunk {i+1}" for i in range(len(chunk_sentiments))]

    def advanced_sentiment_analysis(self, texts, file_paths):
        all_sentiments, all_files, all_chunks = [], [], []

        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(self.sentiment_analysis_chunk, text, file_path): (text, file_path) for text, file_path in zip(texts, file_paths)}
            for future in as_completed(futures):
                chunk_sentiments, file_chunks, chunk_labels = future.result()
                all_sentiments.extend(chunk_sentiments)
                all_files.extend(file_chunks)
                all_chunks.extend(chunk_labels)

        print(f"Completed sentiment analysis for {len(texts)} texts")
        return all_sentiments, all_files, all_chunks

    def plot_sentiment_scores_by_topic(self, sentiments, topics, file_paths):
        df_sentiment = pd.DataFrame({
            'Sentiment Score': sentiments,
            'Topic': topics,
            'File Path': file_paths
        })

        # Convert topics to string for consistent sorting and mapping
        df_sentiment['Topic'] = df_sentiment['Topic'].astype(str)

        # Sort the DataFrame by Topic and then by Sentiment Score within each Topic
        df_sentiment = df_sentiment.sort_values(by=['Topic', 'Sentiment Score']).reset_index(drop=True)

        # Assigning a color to each topic
        unique_topics = df_sentiment['Topic'].unique()
        color_map = {topic: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] for i, topic in enumerate(unique_topics)}
        df_sentiment['Color'] = df_sentiment['Topic'].map(color_map)

        # Plotting
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=df_sentiment.index,
            y=df_sentiment['Sentiment Score'],
            marker_color=df_sentiment['Color'],
            text=df_sentiment['File Path'],
            hoverinfo='text',
        ))

        fig.update_layout(
            title='Sentiment Scores Grouped by Topic',
            xaxis_title='Document Index (Grouped by Topic)',
            yaxis_title='Sentiment Score',
            showlegend=False
        )

        fig.show()

class TopicModeler:
    def __init__(self, num_topics=None):
        self.vectorizer_model = TfidfVectorizer(stop_words='english')
        self.topic_model = BERTopic(vectorizer_model=self.vectorizer_model, language="english", nr_topics=num_topics)
        print(f"Initialized TopicModeler with {num_topics if num_topics else 'default'} topics.")

    def perform_bertopic(self, texts, raw_texts):
        topics, probs = self.topic_model.fit_transform(raw_texts)
        print(f"Completed BERTopic analysis")
        return topics, probs

    def perform_lda_pca(self, texts):
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        lda_model = models.LdaModel(corpus, num_topics=5, random_state=42)
        
        # Create a topic distribution matrix for PCA
        topic_distributions = [lda_model.get_document_topics(doc, minimum_probability=0) for doc in corpus]
        topic_matrix = np.array([[topic[1] for topic in dist] for dist in topic_distributions])
        
        # Apply PCA
        pca = PCA(n_components=2, random_state=42)
        pca_result = pca.fit_transform(topic_matrix)
        
        # Instead of calling `show_topics` directly, let's manually generate labels
        topics = lda_model.show_topics(num_topics=2, num_words=3, formatted=False)
        pca_labels = {
            "x_label": ", ".join([word for word, _ in topics[0][1]]),
            "y_label": ", ".join([word for word, _ in topics[1][1]]) if len(topics) > 1 else ""
        }
       
        print("Completed LDA and PCA")
        return lda_model, pca_result, topic_matrix, pca_labels

    def get_bertopic_pca_axis_labels(self, num_words=3):
        topics = self.topic_model.get_topics()
        sorted_topics = sorted(topics.items(), key=lambda x: -len(x[1]))  # Sort by topic size
        pca_labels = {"x_label": "", "y_label": ""}
        if len(sorted_topics) > 0:
            pca_labels["x_label"] = ", ".join([word for word, _ in sorted_topics[0][1][:num_words]])
        if len(sorted_topics) > 1:
            pca_labels["y_label"] = ", ".join([word for word, _ in sorted_topics[1][1][:num_words]])
        return pca_labels

    def perform_tsne_on_lda(self, topic_matrix):
        tsne_model = TSNE(n_components=2, random_state=42, perplexity=min(30, len(topic_matrix)-1))
        tsne_result = tsne_model.fit_transform(topic_matrix)
        print("Completed t-SNE on LDA")
        return tsne_result

class Plotter:
    def plot_lda_pca(self, pca_result, file_paths, topic_modeler, prefix_colors, model):
        df_pca = pd.DataFrame({'x': pca_result[:, 0], 'y': pca_result[:, 1], 'file_path': file_paths})
        df_pca['Prefix'] = [os.path.basename(fp).split('_')[0] for fp in file_paths]
        df_pca['Color'] = df_pca['Prefix'].map(prefix_colors)

        fig_pca = go.Figure()

        fig_pca.add_trace(go.Scatter(
            x=df_pca['x'],
            y=df_pca['y'],
            mode='markers',
            marker=dict(color=df_pca['Color']),
            text=df_pca['file_path'],
            hoverinfo='text'
        ))

        # Get the correct PCA axis labels
        pca_labels = topic_modeler.get_bertopic_pca_axis_labels(num_words=3)

        # Add a dropdown to select how many top words to display
        fig_pca.update_layout(
            title='PCA of Topic Distribution',
            xaxis_title=f'PCA1: {pca_labels["x_label"]}',
            yaxis_title=f'PCA2: {pca_labels["y_label"]}',
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [
                                {
                                    "xaxis.title.text": f'PCA1: {topic_modeler.get_bertopic_pca_axis_labels(num_words=3)["x_label"]}',
                                    "yaxis.title.text": f'PCA2: {topic_modeler.get_bertopic_pca_axis_labels(num_words=3)["y_label"]}'
                                }
                            ],
                            "label": "Top 3 Words",
                            "method": "relayout"
                        },
                        {
                            "args": [
                                {
                                    "xaxis.title.text": f'PCA1: {topic_modeler.get_bertopic_pca_axis_labels(num_words=4)["x_label"]}',
                                    "yaxis.title.text": f'PCA2: {topic_modeler.get_bertopic_pca_axis_labels(num_words=4)["y_label"]}'
                                }
                            ],
                            "label": "Top 4 Words",
                            "method": "relayout"
                        },
                        {
                            "args": [
                                {
                                    "xaxis.title.text": f'PCA1: {topic_modeler.get_bertopic_pca_axis_labels(num_words=5)["x_label"]}',
                                    "yaxis.title.text": f'PCA2: {topic_modeler.get_bertopic_pca_axis_labels(num_words=5)["y_label"]}'
                                }
                            ],
                            "label": "Top 5 Words",
                            "method": "relayout"
                        }
                    ],
                    "direction": "down",
                    "showactive": True,
                }
            ]
        )

        fig_pca.show()

    def plot_tsne_lda(self, tsne_result, file_paths, prefix_colors):
        df_tsne = pd.DataFrame({'x': tsne_result[:, 0], 'y': tsne_result[:, 1], 'file_path': file_paths})
        df_tsne['Prefix'] = [os.path.basename(fp).split('_')[0] for fp in file_paths]
        df_tsne['Color'] = df_tsne['Prefix'].map(prefix_colors)

        fig_tsne = go.Figure()

        fig_tsne.add_trace(go.Scatter(
            x=df_tsne['x'],
            y=df_tsne['y'],
            mode='markers',
            marker=dict(color=df_tsne['Color']),
            text=df_tsne['file_path'],
            hoverinfo='text'
        ))

        fig_tsne.update_layout(
            title='t-SNE of LDA Topic Distributions',
            xaxis_title=f't-SNE Component 1: {df_tsne["x"].name}',
            yaxis_title=f't-SNE Component 2: {df_tsne["y"].name}'
        )
        fig_tsne.show()

    def plot_bertopic_2d(self, topic_model, topics, probs, raw_texts, sentiments, files, chunks):
        embeddings = topic_model.embedding_model.embedding_model.encode(raw_texts)
        tsne_model = TSNE(n_components=2, random_state=42, perplexity=min(30, len(raw_texts)-1))
        tsne_embeddings = tsne_model.fit_transform(embeddings)
        df_tsne = pd.DataFrame(tsne_embeddings, columns=['x', 'y'])
        df_tsne['topic'] = topics
        df_tsne['sentiment'] = sentiments
        df_tsne['file'] = files
        df_tsne['chunk'] = chunks
        fig_tsne = go.Figure()

        fig_tsne.add_trace(go.Scatter(
            x=df_tsne['x'],
            y=df_tsne['y'],
            mode='markers',
            marker=dict(color=df_tsne['sentiment'], colorscale='RdYlGn', showscale=True),
            text=df_tsne['file'],
            hoverinfo='text'
        ))

        fig_tsne.update_layout(
            title='t-SNE of BERTopic Distribution with Sentiment',
            xaxis_title='t-SNE Component 1',
            yaxis_title='t-SNE Component 2'
        )
        fig_tsne.show()

    def plot_bertopic_wordcloud(self, topic_model):
        topic_freq = topic_model.get_topic_freq()
        for topic in topic_freq['Topic']:
            if topic == -1:
                continue  # Skip outlier topic

            words = topic_model.get_topic(topic)
            word_dict = {word: value for word, value in words}

            if len(word_dict) == 0:
                continue

            wc = WordCloud(
                width=800,
                height=400,
                max_words=10,
                background_color="white",
                colormap="viridis"
            ).generate_from_frequencies(word_dict)
            
            img = wc.to_image()
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            fig = go.Figure(go.Image(source=f'data:image/png;base64,{img_str}'))
            fig.update_layout(title=f'Topic {topic} Word Cloud')
            fig.show()

class WordCloudGenerator:
    def __init__(self, frequency_threshold=0):
        self.frequency_threshold = frequency_threshold
        print(f"Initialized WordCloudGenerator with frequency threshold: {self.frequency_threshold}")

    def generate_wordcloud(self, texts, title, max_words=100, background_color="white", colormap="viridis", max_font_size=None):
        all_words = [word for text in texts for word in text]
        word_freq = Counter(all_words)

        filtered_words = {word: freq for word, freq in word_freq.items() if freq >= len(texts) * self.frequency_threshold}

        if not filtered_words:
            print("No words left after filtering. Skipping word cloud generation.")
            return

        wordcloud = WordCloud(
            width=800,
            height=400,
            max_words=max_words,
            background_color=background_color,
            colormap=colormap,
            max_font_size=max_font_size
        ).generate_from_frequencies(filtered_words)

        img = wordcloud.to_image()
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        fig = go.Figure(go.Image(source=f'data:image/png;base64,{img_str}'))
        fig.update_layout(title=title)
        fig.show()

class TextAnalysisPipeline:
    def __init__(self):
        self.text_processor = TextProcessor()
        self.sentiment_analyzer = None  # Initialize later with user input
        self.topic_modeler = None  # Initialize later with user input
        self.plotter = Plotter()
        self.wordcloud_generator = None  # Initialize later with user input
        print("Initialized TextAnalysisPipeline")

    def execute_pipeline(self, source_type='file', folder_name='texts', dataset_name='ag_news', split='train', temperature=1.0, sample_size=None, frequency_threshold=0, chunk_size=5000, num_topics=None):
        if source_type == 'file':
            texts, file_paths = self.text_processor.load_texts_from_folder(folder_name)
        else:  # Assume source_type is 'ag'
            texts, file_paths = self.text_processor.load_texts_from_dataset(dataset_name, split, sample_size)

        if not texts:
            print("No texts loaded. Exiting.")
            return

        # Initialize SentimentAnalyzer with user-defined temperature
        self.sentiment_analyzer = SentimentAnalyzer(temperature=temperature)

        # Initialize WordCloudGenerator with the given frequency threshold
        self.wordcloud_generator = WordCloudGenerator(frequency_threshold)

        # Initialize TopicModeler with the specified number of topics
        self.topic_modeler = TopicModeler(num_topics=num_topics)

        # Perform analysis in chunks if the dataset is large
        num_chunks = (len(texts) // chunk_size) + 1
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, len(texts))
            chunk_texts = texts[start_idx:end_idx]
            chunk_file_paths = file_paths[start_idx:end_idx]
            print(f"Processing chunk {i+1}/{num_chunks}...")
            self.run_analysis(chunk_texts, chunk_file_paths)

    def run_analysis(self, texts, file_paths):
        # Monitoring system performance
        mem_usage = psutil.virtual_memory().percent
        cpu_usage = psutil.cpu_percent()
        print(f"Memory usage: {mem_usage}% | CPU usage: {cpu_usage}%")

        processed_texts = self.text_processor.preprocess_texts(texts)
        sentiments, files, chunks = self.sentiment_analyzer.advanced_sentiment_analysis(texts, file_paths)
        topics, probs = self.topic_modeler.perform_bertopic(processed_texts, texts)
        lda_model, pca_result, topic_matrix, pca_labels = self.topic_modeler.perform_lda_pca(processed_texts)

        # Custom color mapping based on dataset splits or other criteria
        prefix_colors = {
            'ag': 'royalblue',
            # Add more mappings if needed
        }

        # Plotting with custom colors
        self.sentiment_analyzer.plot_sentiment_scores_by_topic(sentiments, topics, files)
        self.plotter.plot_bertopic_2d(self.topic_modeler.topic_model, topics, probs, texts, sentiments, files, chunks)
        
        # Corrected plot_lda_pca function call
        self.plotter.plot_lda_pca(pca_result, files, self.topic_modeler, prefix_colors, model=lda_model)

        # t-SNE on LDA
        tsne_lda_result = self.topic_modeler.perform_tsne_on_lda(topic_matrix)
        self.plotter.plot_tsne_lda(tsne_lda_result, files, prefix_colors)

        # Plot BERTopic WordCloud
        self.plotter.plot_bertopic_wordcloud(self.topic_modeler.topic_model)

        # Generate and plot Word Cloud for the entire corpus
        self.wordcloud_generator.generate_wordcloud(processed_texts, 'Total Corpus Word Cloud')

def get_user_input():
    # Set the default source type to 'AG' (case sensitive)
    source_type = input("Enter 'file' to load texts from a folder or 'AG' to use the AG News dataset (default is 'AG'): ").strip()
    
    if not source_type or source_type == 'AG':
        source_type = 'AG'
        dataset_name = 'ag_news'
        split = input("Enter the dataset split to use (default is 'train'): ")
        if not split:
            split = 'train'

        # Set the default sample size to 100
        sample_input = input("Enter the number of samples to use (default is 100): ")
        try:
            sample_size = int(sample_input) if sample_input else 100
        except ValueError:
            print("Invalid input for sample size. Using default of 100 samples.")
            sample_size = 100

        folder_name = None  # Not needed for AG dataset
    else:
        folder_name = input("Enter the folder name containing the text files (default is 'texts'): ")
        if not folder_name:
            folder_name = 'texts'
        dataset_name = None  # Not needed for file loading
        split = None
        sample_size = None

    # Check if a GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"GPU available: {'Yes' if torch.cuda.is_available() else 'No'}")

    temp_input = input("Enter the temperature for softmax (default is 1.0): ")
    try:
        temperature = float(temp_input) if temp_input else 1.0
    except ValueError:
        print("Invalid input for temperature. Using default value 1.0.")
        temperature = 1.0

    freq_input = input("Enter the word frequency threshold for word clouds (default is 0): ")
    try:
        frequency_threshold = float(freq_input) if freq_input else 0
    except ValueError:
        print("Invalid input for frequency threshold. Using default value 0.")
        frequency_threshold = 0

    chunk_input = input("Enter the chunk size for processing large datasets (default is 5000): ")
    try:
        chunk_size = int(chunk_input) if chunk_input else 5000
    except ValueError:
        print("Invalid input for chunk size. Using default value 5000.")
        chunk_size = 5000

    # Set the default number of topics to 4
    num_topics_input = input("Enter the number of topics for BERTopic (default is 4): ")
    try:
        num_topics = int(num_topics_input) if num_topics_input else 4
    except ValueError:
        print("Invalid input for number of topics. Using default of 4 topics.")
        num_topics = 4

    return source_type, folder_name, dataset_name, split, temperature, sample_size, frequency_threshold, chunk_size, num_topics

    source_type = input("Enter 'file' to load texts from a folder or 'ag' to use the AG News dataset: ").strip().lower()
    if source_type == 'ag':
        dataset_name = 'ag_news'
        split = input("Enter the dataset split to use (default is 'train'): ")
        if not split:
            split = 'train'

        # Set the default sample size to 100
        sample_input = input("Enter the number of samples to use (default is 100): ")
        try:
            sample_size = int(sample_input) if sample_input else 100
        except ValueError:
            print("Invalid input for sample size. Using default of 100 samples.")
            sample_size = 100

        folder_name = None  # Not needed for AG dataset
    else:
        folder_name = input("Enter the folder name containing the text files (default is 'texts'): ")
        if not folder_name:
            folder_name = 'texts'
        dataset_name = None  # Not needed for file loading
        split = None
        sample_size = None

    # Check if a GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"GPU available: {'Yes' if torch.cuda.is_available() else 'No'}")

    temp_input = input("Enter the temperature for softmax (default is 1.0): ")
    try:
        temperature = float(temp_input) if temp_input else 1.0
    except ValueError:
        print("Invalid input for temperature. Using default value 1.0.")
        temperature = 1.0

    freq_input = input("Enter the word frequency threshold for word clouds (default is 0): ")
    try:
        frequency_threshold = float(freq_input) if freq_input else 0
    except ValueError:
        print("Invalid input for frequency threshold. Using default value 0.")
        frequency_threshold = 0

    chunk_input = input("Enter the chunk size for processing large datasets (default is 5000): ")
    try:
        chunk_size = int(chunk_input) if chunk_input else 5000
    except ValueError:
        print("Invalid input for chunk size. Using default value 5000.")
        chunk_size = 5000

    # Set the default number of topics to 4
    num_topics_input = input("Enter the number of topics for BERTopic (default is 4): ")
    try:
        num_topics = int(num_topics_input) if num_topics_input else 4
    except ValueError:
        print("Invalid input for number of topics. Using default of 4 topics.")
        num_topics = 4

    return source_type, folder_name, dataset_name, split, temperature, sample_size, frequency_threshold, chunk_size, num_topics

    source_type = input("Enter 'file' to load texts from a folder or 'ag' to use the AG News dataset: ").strip().lower()
    if source_type == 'ag':
        dataset_name = 'ag_news'
        split = input("Enter the dataset split to use (default is 'train'): ")
        if not split:
            split = 'train'

        sample_input = input("Enter the number of samples to use (press Enter to use all): ")
        try:
            sample_size = int(sample_input) if sample_input else None
        except ValueError:
            print("Invalid input for sample size. Using all samples.")
            sample_size = None

        folder_name = None  # Not needed for AG dataset
    else:
        folder_name = input("Enter the folder name containing the text files (default is 'texts'): ")
        if not folder_name:
            folder_name = 'texts'
        dataset_name = None  # Not needed for file loading
        split = None
        sample_size = None

    # Check if a GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"GPU available: {'Yes' if torch.cuda.is_available() else 'No'}")

    temp_input = input("Enter the temperature for softmax (default is 1.0): ")
    try:
        temperature = float(temp_input) if temp_input else 1.0
    except ValueError:
        print("Invalid input for temperature. Using default value 1.0.")
        temperature = 1.0

    freq_input = input("Enter the word frequency threshold for word clouds (default is 0): ")
    try:
        frequency_threshold = float(freq_input) if freq_input else 0
    except ValueError:
        print("Invalid input for frequency threshold. Using default value 0.")
        frequency_threshold = 0

    chunk_input = input("Enter the chunk size for processing large datasets (default is 5000): ")
    try:
        chunk_size = int(chunk_input) if chunk_input else 5000
    except ValueError:
        print("Invalid input for chunk size. Using default value 5000.")
        chunk_size = 5000

    num_topics_input = input("Enter the number of topics for BERTopic (default is automatic): ")
    try:
        num_topics = int(num_topics_input) if num_topics_input else None
    except ValueError:
        print("Invalid input for number of topics. Using default automatic setting.")
        num_topics = None

    return source_type, folder_name, dataset_name, split, temperature, sample_size, frequency_threshold, chunk_size, num_topics

if __name__ == "__main__":
    source_type, folder_name, dataset_name, split, temperature, sample_size, frequency_threshold, chunk_size, num_topics = get_user_input()
    pipeline = TextAnalysisPipeline()
    pipeline.execute_pipeline(source_type, folder_name, dataset_name, split, temperature, sample_size, frequency_threshold, chunk_size, num_topics)
