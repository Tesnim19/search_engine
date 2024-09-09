from flask import Flask, render_template, request, jsonify
import re
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# Stopwords and lemmatizer setup
stopwords_list = stopwords.words('english')
english_stopset = set(stopwords.words('english')).union({
    "things", "that's", "something", "take", "don't", "may", "want", "you're",
    "set", "might", "says", "including", "lot", "much", "said", "know",
    "good", "step", "often", "going", "thing", "things", "think",
    "back", "actually", "better", "look", "find", "right", "example",
    "verb", "verbs"
})

# Sample documents and titles
docs = [
    'Ethiopia, known as the cradle of humanity, has a history that stretches back millions of years. '
    'Archaeological findings such as "Lucy," a 3.2 million-year-old hominid skeleton, highlight the country\'s ancient past. '
    'By the 4th century, we had adopted Christianity, making it one of the earliest nations to do so. '
    'The Axumite Empire, which thrived between the 1st and 8th centuries AD, was a significant trading empire, connecting Rome and India. '
    'Ethiopia’s rich history also includes resistance against colonial powers, most notably during the Battle of Adwa in 1896, where our forces defeated the Italian army, '
    'preserving its sovereignty while much of Africa was colonized.',

    'The Blue Nile River, originating from Lake Tana, is one of the two major tributaries of the Nile River, contributing approximately 59% of its flow. '
    'It travels over 1,450 kilometers (900 miles) before joining the White Nile in Khartoum, Sudan, forming the Nile that flows into Egypt. '
    'The river is not only crucial for Ethiopia’s agriculture, supporting the livelihoods of millions, but also holds spiritual significance. '
    'The construction of the Grand Ethiopian Renaissance Dam (GERD) on the Blue Nile has been a game-changer, aiming to produce over 6,450 megawatts of electricity, making it the largest hydropower project in Africa.',

    'Ethiopia is the second most populous nation in Africa, with an estimated population of over 118 million people as of 2023. '
    'The country is a mosaic of over 80 ethnic groups, each with its own distinct language, culture, and traditions. '
    'The Oromo, Amhara, and Tigray are the largest ethnic groups, collectively making up more than 60% of the population. '
    'Ethiopia is known for its cultural diversity, with ancient festivals like Timkat (Epiphany) and Meskel (Finding of the True Cross) attracting millions of participants annually. '
    'The country also boasts a rich musical heritage, with traditional instruments like the krar (lyre) and masenqo (one-stringed fiddle) still popular today.',

    'The Grand Ethiopian Renaissance Dam (GERD) is a flagship project for Ethiopia, representing its aspirations for energy self-sufficiency and economic development. '
    'Located on the Blue Nile River in the Benishangul-Gumuz region, the dam is expected to cost $5 billion USD and generate over 6,450 megawatts of electricity upon completion. '
    'The GERD is seen as a symbol of national pride, with at home and abroad contributing financially to its construction. '
    'However, the project has also been a source of regional tension, particularly with downstream countries like Egypt and Sudan, which rely heavily on the Nile for their water supply. '
    'It has assured these countries that the dam will not significantly reduce water flow, proposing a cooperative framework for water management in the Nile Basin.',

    'Ethiopia is often referred to as the birthplace of coffee, with the legend of Kaldi, a goat herder who discovered coffee in the 9th century, originating from here. '
    'Today, it is Africa’s top coffee producer and the 5th largest in the world, producing over 7 million 60-kilogram bags annually. '
    'Coffee is deeply embedded in culture, with the traditional coffee ceremony being a vital social event. '
    'The ceremony involves roasting green coffee beans, grinding them, and brewing the coffee in a jebena (a traditional clay pot). '
    'Ethiopia’s coffee is known for its diverse flavors, with regions like Sidamo, Yirgacheffe, and Harrar each producing distinct taste profiles ranging from fruity and floral to spicy and wine-like.',

    'Addis Ababa, founded in 1886 by Emperor Menelik II, is the capital city of Ethiopia and home to over 5 million people. '
    'The city, whose name means "New Flower" in Amharic, is the political and diplomatic hub of Africa, hosting the headquarters of the African Union and the United Nations Economic Commission for Africa. '
    'Addis Ababa is a melting pot of cultures, with people from all over the country flocking to the city in search of better opportunities. '
    'The city is also a center for education and research, home to institutions like Addis Ababa University and the Ethiopian Academy of Sciences. '
    'In recent years, Addis Ababa has seen rapid urbanization, with skyscrapers and modern infrastructure reshaping its skyline. '
    'Despite this growth, the city retains its historic charm, with landmarks like the National Museum of Ethiopia, which houses the famous "Lucy" skeleton, and the Holy Trinity Cathedral, a significant site for Ethiopian Orthodox Christians.'
]

titles = [
    'Ethiopian History and Heritage',
    'The Blue Nile River and Its Significance',
    'Population and Cultural Diversity of Ethiopia',
    'The Grand Ethiopian Renaissance Dam',
    'Ethiopian Coffee: The Birthplace of Coffee',
    'Addis Ababa: The Capital City of Ethiopia'
]

# Keywords list (optional)
keywords = [
    'Ethiopia', 'history', 'heritage', 'influence', 'civilization', 'empire',
    'Nile', 'river', 'significance', 'source', 'water', 'flow',
    'population', 'diversity', 'culture', 'tradition', 'language', 'community',
    'dam', 'electricity', 'construction', 'project', 'energy',
    'coffee', 'origin', 'cultivation', 'export', 'beverage',
    'Addis', 'Ababa', 'city', 'capital', 'government'
]

document_clean = []

# Document preprocessing and TF-IDF vectorization
lemmatizer = WordNetLemmatizer()
document_clean = [
    ' '.join([lemmatizer.lemmatize(word) for word in re.sub(r'[^\x00-\x7F]+', ' ', doc)
              .lower().translate(str.maketrans('', '', string.punctuation)).split()
              if word not in english_stopset])
    for doc in docs
]

vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0.002, max_df=0.99, max_features=10000, lowercase=True)
X = vectorizer.fit_transform(document_clean)

app = Flask(__name__)

def get_most_relevant_sentence(query, top_n):
    query_clean = ' '.join([lemmatizer.lemmatize(word) for word in re.sub(r'[^\x00-\x7F]+', ' ', query)
                            .lower().translate(str.maketrans('', '', string.punctuation)).split()
                            if word not in english_stopset])
    
    # Handle case when query is empty
    if not query_clean.strip():
        return [{"message": "Query cannot be empty"}]
    
    query_vec = vectorizer.transform([query_clean]).toarray().flatten()
    most_relevant_sentences = []

    for doc_index, doc in enumerate(docs):
        sentences = doc.split('. ')
        sentence_similarities = []
        
        for sentence in sentences:
            sentence_clean = ' '.join([lemmatizer.lemmatize(word) for word in re.sub(r'[^\x00-\x7F]+', ' ', sentence)
                                       .lower().translate(str.maketrans('', '', string.punctuation)).split()
                                       if word not in english_stopset])
            sentence_vec = vectorizer.transform([sentence_clean]).toarray().flatten()
            if np.linalg.norm(sentence_vec) == 0 or np.linalg.norm(query_vec) == 0:
                similarity = 0.0
            else:
                similarity = np.dot(sentence_vec, query_vec) / (np.linalg.norm(sentence_vec) * np.linalg.norm(query_vec))
            
            # Collect sentences with similarity score > 0.0
            if similarity > 0:
                sentence_similarities.append((similarity, sentence))
        
        # Sort sentences based on similarity and add the most relevant
        if sentence_similarities:
            sentence_similarities = sorted(sentence_similarities, key=lambda x: x[0], reverse=True)
            most_relevant_sentences.append((sentence_similarities[0][0], titles[doc_index], sentence_similarities[0][1]))

    # Sort all the sentences by similarity and return the top N relevant results
    most_relevant_sentences = sorted(most_relevant_sentences, key=lambda x: x[0], reverse=True)[:top_n]
    
    # Return meaningful response if no relevant sentences are found
    if not most_relevant_sentences:
        return [{"message": "No relevant results found for your query."}]
    
    return most_relevant_sentences

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query', '')
    top_n = data.get('top_n', 5)

    try:
        top_n = int(top_n)
    except (ValueError, TypeError):
        top_n = 2  # Default value
    
    # Assuming you have some function to process the search
    results = get_most_relevant_sentence(query, top_n)
    
    # Debugging output
    print(f"Query: {query}")
    print(f"Results: {results}")
    
    if not results:
        return jsonify({"message": "No relevant results found."})
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)

