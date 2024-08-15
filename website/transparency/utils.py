import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import json
from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.metrics import silhouette_score
import time
import matplotlib.pyplot as plt  

nltk.download('wordnet')
nltk.download('punkt')

def get_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises HTTPError for bad responses
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text(separator=' ')
        return text
    except requests.exceptions.RequestException as e:
        print(f"Failed to retrieve the URL: {url}. Error: {e}")
        return ''
    except Exception as e:
        print(f"An error occurred while processing the URL: {url}. Error: {e}")
        return ''

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    words = nltk.word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word.lower()) for word in words if word.isalpha()]
    return ' '.join(lemmatized_words)

def calculate_wcss_and_silhouette(tfidf_matrix, max_clusters=15):
    wcss = []
    silhouette_scores = []
    for i in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(tfidf_matrix)
        wcss.append(kmeans.inertia_)
        score = silhouette_score(tfidf_matrix, kmeans.labels_)
        silhouette_scores.append(score)
    return wcss, silhouette_scores

def plot_elbow_and_silhouette(wcss, silhouette_scores, max_clusters=15):
    clusters_range = range(2, max_clusters + 1)
    
    fig, ax1 = plt.subplots()
    
    color = 'tab:red'
    ax1.set_xlabel('Number of clusters')
    ax1.set_ylabel('WCSS', color=color)
    ax1.plot(clusters_range, wcss, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Silhouette Score', color=color)
    ax2.plot(clusters_range, silhouette_scores, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()
    plt.title('Elbow Method and Silhouette Score')
    plt.show()


def train_model(texts, max_clusters=15):
    # Vectorize the texts using TF-IDF with adjusted parameters
    tfidf = TfidfVectorizer(
        stop_words='english',   # Using standard English stop words
        max_df=0.5,             # Lowering max_df to remove more common words
        min_df=4,               # Raising min_df to focus on more frequent and significant terms
        ngram_range=(1, 2)      # Using bi-grams to capture more context
    )
    tfidf_matrix = tfidf.fit_transform(texts)

    # Calculate WCSS and Silhouette Score for a range of clusters
    wcss, silhouette_scores = calculate_wcss_and_silhouette(tfidf_matrix, max_clusters)

    # Plot the Elbow and Silhouette Score
    plot_elbow_and_silhouette(wcss, silhouette_scores, max_clusters)
    
    # Choose the optimal number of clusters, here I'm just using the value where Silhouette Score is highest
    optimal_clusters = np.argmax(silhouette_scores) + 2  # +2 because range starts at 2

    # Fit KMeans with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    kmeans.fit(tfidf_matrix)

    # Perform PCA
    pca = PCA(n_components=3, random_state=42)  # Increase PCA components to retain more variance
    X_pca = pca.fit_transform(tfidf_matrix.toarray())

    pipeline = make_pipeline(
        TfidfVectorizer(
            stop_words='english',   # Using standard English stop words
            max_df=0.5,
            min_df=4,
            ngram_range=(1, 2)
        ),
        PCA(n_components=3, random_state=42),
        KMeans(n_clusters=optimal_clusters, random_state=42)
    )
    pipeline.fit(texts)
    return pipeline, X_pca, kmeans.labels_

def generate_plots(urls):
    try:
        start_time = time.time()

        base_urls = [
            "https://www.bbc.co.uk/sport/olympics/articles/c1d73g2glr0o",
            "https://www.independent.co.uk/news/uk/home-news/uk-riots-far-right-children-radicilisation-parents-b2593480.html",
            "https://www.independent.co.uk/news/world/middle-east/israel-iran-gaza-war-hamas-hezbollah-attack-airspace-news-b2593718.html",
            "https://www.bbc.co.uk/news/live/cze5n6w06j3t",
            "https://www.bbc.co.uk/news/articles/c5y83kj3wg2o",
            "https://www.bbc.co.uk/news/articles/ce81nelyrnlo",
            "https://www.theguardian.com/politics/article/2024/aug/11/far-right-unchristian-archbishop-of-canterbury-justin-welby-condemns-riots",
            "https://www.theguardian.com/world/article/2024/aug/11/thousands-of-ukrainian-troops-aim-to-destabilise-russia-with-kursk-incursion",
            "https://www.theguardian.com/politics/article/2024/aug/11/rioting-swift-justice-real-motivations-behind-uk-rampage",
            "https://news.sky.com/story/boardmasters-mother-describes-festival-chaos-as-she-helped-boy-delirious-with-pain-after-crowd-surge-13195073",
            "https://www.npr.org/2024/08/11/nx-s1-5070566/trump-news-conference",
            "https://www.npr.org/2024/08/09/nx-s1-5060012/kamala-harris-tim-walz-campaign-democrats",
            "https://www.allsides.com/story/facts-and-fact-checking-walz-misspoke-military-experience-2018-video-harris-campaign-says",
            "https://www.allsides.com/story/facts-and-fact-checking-walz-misspoke-military-experience-2018-video-harris-campaign-says",
            "https://www.allsides.com/story/housing-and-homelessness-mortgage-rates-hit-lowest-point-2023",
            "https://www.allsides.com/story/sports-algerias-imane-khelif-wins-boxing-gold-amid-eligibility-controversy",
            "https://dailycaller.com/2024/08/08/garm-censoring-conservatives-disbands-lawsuits/#google_vignette",
            "https://www.pinkun.com/sport/opinion/24513489.norwich-city-fans-react-mark-attanasio-ownership-news/",
            "https://www.pinkun.com/sport/norwich-city/24512609.norwich-city-marseille-target-jon-rowe-train-u21s/",
            "https://www.pinkun.com/sport/interviews/24510946.jonathan-rowe-no-show-not-factor-poor-oxford-showing/",
            "https://www.pinkun.com/sport/norwich-city/24511388.six-things-might-missed-oxford-united-2-0-city/",
            "https://www.pinkun.com/sport/norwich-city/24512973.norwich-city-mark-attanasio-group-take-majority-control/",
            "https://www.pinkun.com/sport/norwich-city/24513780.norwich-city-davitt-attanasio-delia-wynn-jones-news/",
            "https://www.bbc.co.uk/sport/olympics/articles/c15gzzx1q5zo",
            "https://www.bbc.co.uk/news/articles/cx2lmr29ygjo",
            "https://www.bbc.co.uk/news/articles/cp8nqqz30pmo",
            "https://www.bbc.co.uk/news/articles/cd735zvg1q9o",
            "https://www.bbc.co.uk/news/articles/c4gex1rl6q5o",
            "https://www.bbc.co.uk/news/articles/cn0lx2xgn55o",
            "https://www.bbc.co.uk/news/articles/clyl3yg7wzzo",
            "https://www.bbc.co.uk/news/articles/cx2yylgze4ro"
        ]

        all_urls = base_urls + urls
        input_url = urls[0]  # Assuming only one URL is submitted
        texts = []
        valid_urls = []  # Track only valid URLs
        for url in all_urls:
            text = get_text_from_url(url)
            if text:
                lemmatized_text = lemmatize_text(text)
                texts.append(lemmatized_text)
                valid_urls.append(url)  # Add to valid URLs only if successful
            else:
                print(f"Failed to extract text from URL: {url}")

        if not texts:
            raise ValueError("No valid text could be retrieved from the provided URLs.")
        
        if len(texts) != len(valid_urls):
            raise ValueError("Inconsistent data lengths, unable to generate plots.")

        pipeline, X_pca, labels = train_model(texts)

        df = pd.DataFrame({
            'PCA1': X_pca[:, 0],
            'PCA2': X_pca[:, 1],
            'PCA3': X_pca[:, 2],
            'Cluster': labels,
            'URL': valid_urls,
            'Highlight': ['Input URL' if url == input_url else f'Cluster {label}' for url, label in zip(valid_urls, labels)]
        })

        fig_2d = px.scatter(df, x='PCA1', y='PCA2', color='Highlight', hover_data=['URL'], title="2D PCA of News Articles (PCA1 vs PCA2)")
        fig_3d = px.scatter_3d(df, x='PCA1', y='PCA2', z='PCA3', color='Highlight', hover_data=['URL'], title="3D PCA of News Articles (PCA1, PCA2, PCA3)")

        graphJSON_2d = json.loads(fig_2d.to_json())
        graphJSON_3d = json.loads(fig_3d.to_json())

        cluster_texts = " ".join(texts[i] for i in range(len(labels)) if labels[i] == labels[-1])
        wordcloud = WordCloud(max_words=100, background_color='white').generate(cluster_texts)
        wordcloud.to_file('transparency/static/plots/wordcloud.png')
        
        result = {
            'graphData2D': graphJSON_2d,
            'graphData3D': graphJSON_3d
        }
        print(f"Generated graphData: {json.dumps(result, indent=2)}")

        return result
    except Exception as e:
        print(f"Error generating plots: {e}")
        return None
    
def plot_elbow_and_silhouette(wcss, silhouette_scores, max_clusters=15):
    clusters_range = range(2, max_clusters + 1)
    
    fig, ax1 = plt.subplots()  # This should work correctly with plt.subplots()
    
    color = 'tab:red'
    ax1.set_xlabel('Number of clusters')
    ax1.set_ylabel('WCSS', color=color)
    ax1.plot(clusters_range, wcss, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Silhouette Score', color=color)
    ax2.plot(clusters_range, silhouette_scores, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()
    plt.title('Elbow Method and Silhouette Score')
    plt.show()
