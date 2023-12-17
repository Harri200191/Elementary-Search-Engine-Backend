import os 
import warnings 
import json
import flask_cors, flask
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from diffusers import AutoPipelineForText2Image
import torch
from io import BytesIO
import re
import base64
from PIL import Image 
from dotenv import load_dotenv
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import validators

# Load environment variables from .env file
load_dotenv()

nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download('stopwords')
# Initialize Word_Net_Lemmatizer
lemmatizer = WordNetLemmatizer()
warnings.filterwarnings('ignore')


torch.cuda.empty_cache()
pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")

def load_data_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

app = Flask(__name__)
CORS(app)

# Function that takes in content, preprocesses it, 
# and converts it to a list of words
def pre_process_string(content):
    # Remove \n and \t
    content = content.replace('\n', ' ')
    content = content.replace('\t', ' ')
    # Remove all non-characters
    content = re.sub(r'[^a-zA-Z\s]', ' ', content)
    # Remove multiple spaces
    content = re.sub(r'\s+', ' ', content)
    # Convert all characters to lowercase
    content = content.lower()
    # Convert the title into a list of words
    content = content.split()
    # Remove one and two character words
    content = [word for word in content if len(word) > 2]
    # Remove stop_words using nltk
    content = [word for word in content if not word in stopwords.words('english')]
    return content
    
# Function that takes in a list of words and adds them to the lexicon
def build_lexicon(words, lexicon):
    # Build the lexicon
    new_words = []
    # Look through the words
    for word in words:
        # Lemmatize the word
        word = lemmatizer.lemmatize(word)
        # if that word is not already in lexicon
        if word not in lexicon and word not in new_words:
            # Then add it
            new_words.append(word)
    lexicon.extend(new_words)
    return lexicon

# Function to build forward index from raw articles
def build_forward_index(articles):
    # initialize forward_index
    forward_index = dict()

    #initialize documents
    docs = dict()

    # Load the already existing forward_index
    try:
        data = load_data_from_json(r"Files\forward_index.json")
    except:
        with open(r"Files\forward_index.json", "w") as file:
            json.dump(dict(), file)
        data = load_data_from_json(r"Files\forward_index.json")
        
    # Load the lexicon
    try:
        lexicon = load_data_from_json(r"Files\lexicon.json")
    except:
        with open(r"Files\lexicon.json", "w") as file:
            json.dump(list(), file)
        lexicon = load_data_from_json(r"Files\lexicon.json")

    # Load the documents
    try:
        documents = load_data_from_json(r"Files\documents.json")
    except:
        with open(r"Files\documents.json", "w") as file:
            json.dump(dict(), file)
        documents = load_data_from_json(r"Files\documents.json")
        
    num_articles = len(documents)
    
    # Extract all urls currently indexed
    try:
        article_urls = [article['url'] for article in documents.values()]
    except:
        article_urls = []
        
    # For each article
    for article in articles:
        # if article is not already forward indexed
        if article['url'] not in article_urls:
            # Pre-process the title and content
            title_words = pre_process_string(article['title'])
            content_words = pre_process_string(article['content'])
            # Update the lexicon
            lexicon = build_lexicon(title_words + content_words, lexicon)
            # Lemmatize the words in content and title
            content_words = [lemmatizer.lemmatize(word) for word in content_words]
            title_words = [lemmatizer.lemmatize(word) for word in title_words]
            # Convert the words in title and content to their respective indexes
            content_ids = [lexicon.index(word) for word in content_words]
            title_ids = [lexicon.index(word) for word in title_words]
            # Count the frequencies of words
            frequency = Counter((title_ids * 10) + content_ids)
            forward_index[num_articles] = frequency
            docs[num_articles] = {'title': article['title'], 'url': article['url']}
            # Add the url to the article
            article_urls.append(article['url'])
            num_articles += 1
    data.update(forward_index)
    documents.update(docs)
    # Update the lexicon json file
    with open(r"Files\lexicon.json", "w") as file:
        json.dump(lexicon, file)
    # Update the forward_index json file
    with open(r"Files\forward_index.json", "w") as file:
        json.dump(data, file)
    # Update the documents json file
    with open(r"Files\documents.json", "w") as file:
        json.dump(documents, file)

def build_inverted_index_with_barrels():
    # Load the forward index
    try:
        forward_index = load_data_from_json(r"Files\forward_index.json")
    except:
        with open(r"Files\forward_index.json", "w") as file:
            json.dump(dict(), file)
        forward_index = load_data_from_json(r"Files\forward_index.json")

    barrels = []
    barrel_files = os.listdir(r"Files\Barrels")
    # Load all barrels that currently exist
    for barrel in barrel_files:
        barrels.append(load_data_from_json(os.path.join(r"Files\Barrels", barrel)))

    # Iterate through all articles in the forward_index
    for doc_id, data in forward_index.items():
        # Look at all words in an article
        for word_id in data:
            # Calculate the barrel number for that word
            barrel_no = int(word_id) // 10000
            barrel_filename = f"barrel_{barrel_no}.json"
            
            # Check if that barrel exists, if not then create it
            barrel_path = os.path.join(r"Files\Barrels", barrel_filename)
            if not os.path.exists(barrel_path):
                with open(barrel_path, "w") as file:
                    json.dump(dict(), file)
                # Load the newly created barrel
                barrels.append(load_data_from_json(barrel_path))
                barrel_files.append(barrel_filename)
            # update the word_id
            word_id_new = int(word_id) % 10000
            # If that word is not already in that barrel
            if word_id_new not in barrels[barrel_no]:
                # Then create a dict at that word_id
                barrels[barrel_no][word_id_new] = dict()
            # And add the doc_id for that word along with frequency if it is not already there
            if doc_id not in barrels[barrel_no][word_id_new]:
                barrels[barrel_no][word_id_new].update({doc_id: data[word_id]})

    # Sort the barrels
    for i, barrel in enumerate(barrels):
        sorted_data = {}
        for outer_key, inner_dict in barrel.items():
            sorted_inner = dict(sorted(inner_dict.items(), key=lambda x: x[1], reverse=True))
            sorted_data[outer_key] = sorted_inner
            barrels[i] = sorted_data
    # Update all barrels
    i = 0
    for barrel in barrel_files:
        with open(os.path.join(r"Files\Barrels", barrel), "w") as file:
            json.dump(barrels[i], file)
            i += 1
    
    # Clear the forward_index
    with open(r"Files\forward_index.json", "w") as file:
        json.dump(dict(), file)


def rank_results(search_result): 
     # Rank these documents
    # Sort the dictionary by values (descending order)
    sorted_tuples = sorted(search_result.items(), key=lambda x: x[1], reverse=True)
    
    # Convert the sorted list of tuples back to a dictionary
    ranked_result = dict(sorted_tuples)
    # Extract the article ids
    ranked_articles = ranked_result.keys()
    ranked_articles = list(ranked_articles)
    
    return ranked_articles

def add_content(data, new_article):
    article_id = str(len(data["index"]))
    data["index"][article_id] = len(data["index"])
    data["source"][article_id] = new_article[0]
    data["title"][article_id] = new_article[1]
    data["content"][article_id] = new_article[2]

    return data

barrels = []
barrel_files = os.listdir(r"Files\Barrels")
# Load all barrels that currently exist
for barrel in barrel_files:
    barrels.append(load_data_from_json(os.path.join(r"Files\Barrels", barrel)))
    
# Load lexicon
lexicon = load_data_from_json(r"Files\lexicon.json")
# Load the documents
documents = load_data_from_json(r"Files\documents.json")

app = Flask(__name__)
CORS(app)

@app.route("/search_1", methods=["GET"], endpoint='single_word_search')
def single_word_search():
    word = request.args.get('word')

    # Lemmatize the word
    word = lemmatizer.lemmatize(word)
        
    # Find the id of the word in lexicon
    try:
        word_id = lexicon.index(word)
        # Calculate the barrel of the word
        barrel_no = word_id // 10000
        # Update the word_id
        word_id = word_id % 10000
        # Find out in which documents does the word appear
        search_result = barrels[barrel_no][str(word_id)]
    except:
        search_result = None
    
    if search_result is None: 
        return jsonify(article_ids=[], titles=[], urls=[])

    article_ids = list(search_result.keys())
    titles = [documents[article]['title'] for article in article_ids]
    urls = [documents[article]['url'] for article in article_ids]
    
    json_response = jsonify(article_ids=article_ids, titles=titles, urls=urls)

    return json_response

@app.route("/search_2", methods=["GET"], endpoint='multi_word_search')
def multi_word_search(): 
    query = request.args.get('word')
    result = []

    # Preprocess the query
    words = pre_process_string(query)

    # Remove those words that are not in lexicon
    words = [word for word in words if word in lexicon]
    # Convert each word to its word_id
    word_ids = [lexicon.index(word) for word in words]
    # Calculate barrel_no of each word and its index in that barrel
    barrel_nos = [word_id // 10000 for word_id in word_ids]
    word_ids = [word_id % 10000 for word_id in word_ids]

    # Check the first word
    if word_ids:
        result = barrels[barrel_nos[0]][str(word_ids[0])]
        # Check the rest of the words
        for i, word_id in enumerate(word_ids[1:], start = 1):
            # Produce the result for current word
            current_result = barrels[barrel_nos[i]][str(word_id)]
            # Include those articles that are also in the result of current word
            result.update({d:result[d]+current_result[d] for d in result.keys() if d in current_result.keys()})

    if result is None:
        return jsonify(article_ids=[], titles=[], urls=[])
    
    # rank the results
    result = rank_results(result)

    article_ids = result
    titles = [documents[article]['title'] for article in article_ids]
    urls = [documents[article]['url'] for article in article_ids]

    json_response = jsonify(article_ids=article_ids, titles=titles, urls=urls)

    return json_response


@app.route("/gen", methods=["GET"], endpoint='genai_tool')
def genai_tool():
    word = request.args.get('word') 
    image = pipe(prompt=word, num_inference_steps=1, guidance_scale=0.0).images[0]

    image2 = image.convert("RGB")
 
    image_bytes_io = BytesIO()
    image2.save(image_bytes_io, format="PNG")
    image_bytes = image_bytes_io.getvalue()
    
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')  

    json_response = {
        'word': word,
        'image': image_base64
    }

    return jsonify(json_response)

UPLOAD_FOLDER = 'Files\\Uploads'  
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/add", methods=["POST"], endpoint='add_content')
def add_content():
    title = request.form.get('title')
    url = request.form.get('url')
    content = request.form.get('content')
    file = request.files.get('file') 
 
    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)

    global barrels, lexicon, documents
    # Check if a file is uploaded
    if file:
        try:
            # Load the file
            data = load_data_from_json(filename) 
        except:
            return jsonify({"message": "Error loading file"}), 500
    else:
        # Check if the url, title and content are correct
        if url and title and content:
            # Validate the url
            if validators.url(url) != True:
                return jsonify({"message": "Please provide a valid url"}), 400
            # Load the data
            data = [{"title":title, "content":content, "url":url}]
        else:
            return jsonify({"message": "Please provide a file or url, title and content"}), 400

    # build forward and inverted index on it
    try:
        build_forward_index(data)
    except:
        return jsonify({"message": "Error building forward index"}), 500
    build_inverted_index_with_barrels()

    # Reload barrels, documents, and lexicon
    barrel_files = os.listdir(r"Files\\Barrels")
    # Load all barrels that currently exist
    for i, barrel in enumerate(barrel_files):
        try:
            barrels[i] = (load_data_from_json(os.path.join(r"Files\\Barrels", barrel)))
        except:
            barrels.append(load_data_from_json(os.path.join(r"Files\\Barrels", barrel)))
        
    # Load lexicon
    lexicon = load_data_from_json(r"Files\\lexicon.json")
    # Load the documents
    documents = load_data_from_json(r"Files\\documents.json")
    
    return jsonify({"message": "Successfully added content"}), 200

if __name__ == "__main__":
    app.run(debug=False)