import os
import re
import nltk
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bertopic import BERTopic
from collections import Counter
from wordcloud import WordCloud
import logging
import webbrowser
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import psutil

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set base directory paths
base_dir = "/Users/henryhodges/Documents/Documents - Henry’s MacBook Air/Year 2/Startup/Transparency-2.0/website/BERTtopic"
wordcloud_dir = os.path.join(base_dir, "wordclouds")
os.makedirs(wordcloud_dir, exist_ok=True)  # Ensure the directory exists

# ==============================================
# Base Text Processing  
# ==============================================
class BaseTextProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        return text.strip()

    def tokenize_text(self, text):
        return word_tokenize(text.lower())

    def filter_tokens(self, tokens):
        return [word for word in tokens if word.isalpha() and word not in self.stop_words]

    def get_text_from_url(self, url):
        headers = {'User-Agent': 'Mozilla/5.0'}
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text(separator=' ')
            text = self.clean_text(text)
            return text
        except requests.exceptions.RequestException as e:
            print(f"Failed to retrieve the URL: {url}. Error: {e}")
            return ''
        except Exception as e:
            print(f"An error occurred while processing the URL: {url}. Error: {e}")
            return ''

    def load_texts_from_dataset(self, dataset_name='ag_news', split='train', sample_size=None, url=None):
        try:
            dataset = load_dataset(dataset_name, split=split)
            print(f"Loaded {len(dataset)} records from the {dataset_name} dataset.")
            if sample_size and sample_size < len(dataset):
                dataset = dataset.shuffle(seed=42).select(range(sample_size))
                print(f"Sampled {sample_size} records from the dataset.")
            texts = [record['text'] for record in dataset]
            file_paths = [f"{dataset_name}_{split}_article_{i}.txt" for i in range(len(texts))]

            if url:
                url_text = self.get_text_from_url(url)
                if url_text:
                    texts.append(url_text)
                    file_paths.append(url)
                    print(f"Appended text from URL: {url}")

            return texts, file_paths
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            return [], []

    def load_texts_from_folder(self, folder_path, url=None):
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

        if url:
            url_text = self.get_text_from_url(url)
            if url_text:
                texts.append(url_text)
                file_paths.append(url)
                print(f"Appended text from URL: {url}")

        print(f"Loaded {len(texts)} text files from {folder_path}")
        return texts, file_paths


# ==============================================
# BERT topic text processor
# ==============================================
class BERTopicTextProcessor(BaseTextProcessor):
    def __init__(self, use_tokenization=True):
        super().__init__()
        self.use_tokenization = use_tokenization
        print(f"Initialized BERTopicTextProcessor with tokenization {'enabled' if self.use_tokenization else 'disabled'}.")

    def preprocess_texts_for_bertopic(self, texts):
        processed_texts = []
        for text in texts:
            text = self.clean_text(text)
            text = text.lower()
            if self.use_tokenization:
                tokens = word_tokenize(text)
                filtered_tokens = self.filter_tokens(tokens)  # Apply stopword filtering
                processed_text = ' '.join(filtered_tokens)
            else:
                processed_text = text
            processed_texts.append(processed_text)
        return processed_texts


# ==============================================
# BERT topic analysis 
# ==============================================
class BERTTopicProcessor:
    def __init__(self, num_topics=None, max_df=0.95):
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.topic_model = BERTopic(embedding_model=embedding_model, nr_topics=num_topics)
            self.images_dir = wordcloud_dir  # Set the images directory to the wordclouds folder
            self.stop_words = set(stopwords.words('english'))  # Initialize stopwords
            print(f"Initialized BERTopicProcessor with {num_topics if num_topics else 'default'} topics and max_df={max_df}.")

    def perform_bertopic(self, raw_texts):
        try:
            # Debugging: Print the number of documents and their types
            print(f"Number of documents being processed: {len(raw_texts)}")
            for i, text in enumerate(raw_texts):
                if not isinstance(text, str):
                    print(f"Non-string element found at index {i}: {type(text)} with value: {text}")
                    raise ValueError("All inputs to BERTopic must be strings.")
                if len(text.strip()) == 0:
                    print(f"Empty string found at index {i}")

            topics, probs = self.topic_model.fit_transform(raw_texts)
            print(f"Generated {len(set(topics)) - 1} topics.")  # Log the number of generated topics
            return topics, probs
        except Exception as e:
            logger.error(f"Error during BERTopic analysis: {e}")
            return [], []

    def plot_bertopic_wordcloud(self, processed_texts, corpus_wordcloud=True, word_count=10, image_size=(400, 400)):
            topic_freq = self.topic_model.get_topic_freq()
            valid_topics = topic_freq[topic_freq['Topic'] != -1]
            image_paths = []

            if valid_topics.empty:
                logger.warning("No valid topics found. Word clouds will not be generated.")
                return

            for topic in valid_topics['Topic']:
                words = self.topic_model.get_topic(topic)
                word_dict = {word: value for word, value in words[:word_count] if word not in self.stop_words}

                logger.info(f"Generating word cloud for Topic {topic} with {len(word_dict)} words.")
                if len(word_dict) == 0:
                    logger.warning(f"Topic {topic} has no words to display. Skipping word cloud generation.")
                    continue

                try:
                    wc = WordCloud(
                        width=image_size[0],
                        height=image_size[1],
                        max_words=word_count,
                        background_color="white",
                        colormap="viridis",
                        stopwords=self.stop_words  # Pass the stopwords here
                    ).generate_from_frequencies(word_dict)

                    image_path = os.path.join(self.images_dir, f"topic_{topic}.png")
                    wc.to_file(image_path)
                    image_paths.append(f"wordclouds/topic_{topic}.png")  # Relative path for HTML
                except Exception as e:
                    logger.error(f"Failed to generate or save word cloud for Topic {topic}: {e}")

            if corpus_wordcloud:
                all_words = [word for text in processed_texts for word in text.split()]
                word_freq = Counter(all_words)

                logger.info(f"Generating corpus word cloud with {len(word_freq)} unique words.")
                if not word_freq:
                    logger.warning("Corpus word cloud generation skipped due to lack of words.")
                    return

                try:
                    wc_corpus = WordCloud(
                        width=image_size[0],
                        height=image_size[1],
                        max_words=word_count,
                        background_color="white",
                        colormap="viridis",
                        stopwords=self.stop_words  # Pass the stopwords here
                    ).generate_from_frequencies(word_freq)

                    image_path = os.path.join(self.images_dir, "total_corpus.png")
                    wc_corpus.to_file(image_path)
                    image_paths.append("wordclouds/total_corpus.png")
                except Exception as e:
                    logger.error(f"Failed to generate or save corpus word cloud: {e}")

            self._generate_wordcloud_html(image_paths)

    def _generate_wordcloud_html(self, image_paths):
        try:
            html_content = """
            <html>
            <head>
                <title>BERTopic Word Clouds</title>
                <style>
                    body { font-family: Arial, sans-serif; }
                    .wordcloud-container { display: flex; flex-wrap: wrap; justify-content: space-around; margin: 20px 0; }
                    .wordcloud-box { flex: 1 0 30%; margin: 20px; text-align: center; }
                    .wordcloud-box img { width: 100%; }
                </style>
            </head>
            <body>
                <h1>BERTopic Word Clouds</h1>
                <div class="wordcloud-container">
            """
            for img_path in image_paths:
                topic_number = img_path.split("_")[1]  # Extract the topic number from the filename
                html_content += f"""
                <div class="wordcloud-box">
                    <img src="{img_path}" alt="Word Cloud">
                    <p>Topic {topic_number}</p>
                </div>
                """
            html_content += """
                </div>
            </body>
            </html>
            """
            html_file_path = os.path.join(base_dir, "wordclouds.html")  # Save HTML in the base directory
            with open(html_file_path, "w") as html_file:
                html_file.write(html_content)
            logger.info(f"Word cloud HTML file generated at {html_file_path}. Opening in browser...")
            webbrowser.open(f"file://{html_file_path}")
        except Exception as e:
            logger.error(f"Error generating word cloud HTML file: {e}")

    def list_documents_by_topic(self, raw_texts, topics):
        """ List documents by their assigned topic. """
        documents_by_topic = {}
        for idx, topic in enumerate(topics):
            if topic not in documents_by_topic:
                documents_by_topic[topic] = []
            documents_by_topic[topic].append(raw_texts[idx])
        
        # Display or return the documents grouped by topic
        for topic, docs in documents_by_topic.items():
            print(f"Topic {topic}: {len(docs)} documents")
            for doc in docs:
                print(f"  - {doc[:100]}...")  # Print first 100 characters of each document for a quick preview
        return documents_by_topic

    def process_and_combine_chunks(self, texts, file_paths, chunk_size):
        num_chunks = (len(texts) // chunk_size) + (1 if len(texts) % chunk_size != 0 else 0) # calculate number of chunks

        all_topics = []
        all_probs = [] # initialize arrays

        for i in range(num_chunks): # loops through all chunks
            start_idx = i * chunk_size 
            end_idx = min(start_idx + chunk_size, len(texts)) # range of data for current chunk
            chunk_texts = texts[start_idx:end_idx]
            chunk_file_paths = file_paths[start_idx:end_idx] # extract current chunk of data

            if len(chunk_texts) == 0:
                print(f"Skipping empty chunk {i+1}/{num_chunks}...") # no empty chunks round here dawg
                continue

            print(f"Processing chunk {i+1}/{num_chunks}...")
            topics, probs = self.run_analysis(chunk_texts, chunk_file_paths, special_index=start_idx if url else None) # run analysis on the chunks

            # Accumulate topics and probabilities from all chunks
            all_topics.extend(topics)
            all_probs.extend(probs) # combines topics and probabilities (for document dist.)

        return all_topics, all_probs


# ==============================================
# Control centre
# ==============================================
class TextAnalysisPipeline:
    def __init__(self):
        self.text_processor = BERTopicTextProcessor()
        self.bert_topic_processor = BERTTopicProcessor()

    def execute_pipeline(self, dataset_name='ag_news', split='train', sample_size=None, chunk_size=5000, num_topics=None, max_df=0.95, url=None):
        # Load texts from the dataset
        texts, file_paths = self.text_processor.load_texts_from_dataset(dataset_name, split, sample_size, url=url)

        if not texts:
            print("No texts loaded. Exiting.")
            return None

        # Process and combine all chunks using the method in BERTTopicProcessor
        all_topics, all_probs = self.bert_topic_processor.process_and_combine_chunks(texts, file_paths, chunk_size)

        # Accumulate processed texts for word cloud generation
        all_processed_texts = self.text_processor.preprocess_texts_for_bertopic(texts)

        # After all chunks are processed, generate word clouds with combined data
        self.bert_topic_processor.plot_bertopic_wordcloud(all_processed_texts)

        # Now call list_documents_by_topic with the full dataset
        documents_by_topic = self.bert_topic_processor.list_documents_by_topic(texts, all_topics)
        
        return all_topics, all_probs

    def run_analysis(self, texts, file_paths, special_index=None):
        mem_usage = psutil.virtual_memory().percent
        cpu_usage = psutil.cpu_percent()
        print(f"Memory usage: {mem_usage}% | CPU usage: {cpu_usage}%")

        # Preprocessing texts for BERTopic
        processed_texts_for_bertopic = self.text_processor.preprocess_texts_for_bertopic(texts)

        # Validate that all processed texts are strings
        if not all(isinstance(text, str) for text in processed_texts_for_bertopic):
            raise ValueError("All inputs to BERTopic must be strings.")

        try:
            # Perform BERTopic analysis
            topics, probs = self.bert_topic_processor.perform_bertopic(processed_texts_for_bertopic)
        except Exception as e:
            print(f"Error during BERTopic analysis: {e}")
            topics, probs = [], []  # Default to empty lists if an error occurs

        return topics, probs

if __name__ == '__main__':
    # Ensure NLTK resources are downloaded
    nltk.download('stopwords')
    nltk.download('punkt')
    
    # Get URL input from the user
    url = input("Enter a URL to include in the analysis (optional): ").strip()
    
    # Define parameters for the analysis
    dataset_name = 'ag_news'  # Dataset to load
    split = 'train'  # Dataset split
    sample_size = 5000  # Number of samples to load from the dataset
    chunk_size = 1000  # Size of text chunks for processing
    num_topics = None  # Number of topics for BERTopic (None for automatic)
    max_df = 0.95  # Max document frequency for topic modeling
    
    # Initialize and run the text analysis pipeline
    pipeline = TextAnalysisPipeline()
    results = pipeline.execute_pipeline(
        dataset_name=dataset_name,
        split=split,
        sample_size=sample_size,
        chunk_size=chunk_size,
        num_topics=num_topics,
        max_df=max_df,
        url=url
    )
    
    if results:
        print("Analysis completed. Listing documents by topic.")
        bert_topic_processor = pipeline.bert_topic_processor
        topics, _ = results
        if topics:
            documents_by_topic = bert_topic_processor.list_documents_by_topic(
                pipeline.text_processor.load_texts_from_dataset(dataset_name, split, sample_size)[0], topics
            )
        else:
            print("No topics were generated during the analysis.")