# Fake News Detection System 📰

A robust and intuitive system designed to combat misinformation by classifying news articles as "Real" or "Fake" using Natural Language Processing (NLP) and Machine Learning. This project features a high-accuracy classification model and an interactive web application for real-time analysis.

***

## Features ✨

* **High-Accuracy Classification:** Employs a **Multinomial Naive Bayes** classifier that achieves **93.17% accuracy** on the test dataset.
* **Advanced NLP Techniques:** Uses **TF-IDF** and **Count Vectorizer** for effective feature extraction from textual data, turning unstructured text into meaningful vectors for the model.
* **Real-Time Analysis:** Integrates with a live news API to fetch and analyze the credibility of the latest news stories as they break.
* **Interactive Web Interface:** A user-friendly web app built with **Streamlit** allows anyone to paste news text or a URL for an instant credibility prediction.
* **Model Insights:** The application provides clear visualizations and explanations for the model's prediction, helping users understand the "why" behind the classification.

***

## How It Works 🤖

The system follows a straightforward yet powerful pipeline:

1.  **Data Input:** The user provides news content either by pasting the text directly or by entering a URL. The system can also fetch news from a live API.
2.  **Text Preprocessing:** The raw text is cleaned by removing punctuation, stopwords, and converting all text to lowercase.
3.  **Feature Extraction:** The cleaned text is transformed into a numerical representation using the TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer.
4.  **Prediction:** The pre-trained Multinomial Naive Bayes model takes the numerical vector as input and outputs a prediction ('Real' or 'Fake') along with a confidence score.
5.  **Visualization:** The result is displayed in the Streamlit application in a clear and understandable format.

***

## Tech Stack 🛠️

* **Language:** Python 3.8+
* **Machine Learning:** Scikit-learn
* **NLP:** NLTK, Scikit-learn
* **Web Framework:** Streamlit
* **Data Handling:** Pandas, NumPy
* **API Integration:** Requests

***

## Getting Started 🚀

Follow these instructions to get a local copy up and running.

### Prerequisites

* Python 3.8 or higher
* pip package manager

### Installation

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/your-username/fake-news-detection.git](https://github.com/your-username/fake-news-detection.git)
    cd fake-news-detection
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

4.  **Set up your API Key (if applicable):**
    If you are using a news API, create a `.env` file in the root directory and add your key:
    ```
    NEWS_API_KEY="YOUR_API_KEY_HERE"
    ```

5.  **Run the Streamlit application:**
    ```sh
    streamlit run app.py
    ```
    Navigate to `http://localhost:8501` in your web browser to see the application live!

***

## Model Performance 📈

The core of this system is a **Multinomial Naive Bayes** classifier. It was trained on a labeled dataset of real and fake news articles and evaluated on a hold-out test set.

* **Accuracy:** **93.17%**
* **Key Features:** The model effectively learns to distinguish between the linguistic patterns, vocabulary, and tones commonly found in legitimate journalism versus those typical of fabricated news.

***

