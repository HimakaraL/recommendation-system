from flask import Flask, request, render_template
import pandas as pd
from flask_sqlalchemy import SQLAlchemy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


app = Flask(__name__)

#Reading the dataset
top_animes = pd.read_csv('./models/top_anime.csv')
train_data = pd.read_csv('./AnimeDataWithTags.csv')

#Database Configuration
app.secret_key = '72fade5da4a96210901796a122f06a9e'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/rec_sys'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

#model class for signup
class Signup(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)

#model class for login
class Login(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)

#Model class for recommendation
#Function

def content_based_recommendations(train_data, item_name, top_n=10):
    #Checking for the name to exist in the dataset
    if item_name not in train_data['Name'].values:
        print(f"Item '{item_name}' not found in the dataset.")
        return pd.DataFrame()

    #Create a TF_idf vectorizer for descriptions
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    #Apply tf-idf
    tfidf_matrix_content = tfidf_vectorizer.fit_transform(train_data['Tags'])

    #Calculate cosine similarity
    cosine_similarities_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)

    #find item index
    item_index = train_data[train_data['Name'] == item_name].index[0]

    #cosine similarity scores
    similar_items = list(enumerate(cosine_similarities_content[item_index]))

    #Sort similar items by desc order
    similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)

    #Get the most similar items except item itself
    top_similar_items = similar_items[1:top_n+1]

    #Get the indices of top similar items
    recommended_item_indices = [x[0] for x in top_similar_items]

    #Get the details
    recommended_item_details = train_data.iloc[recommended_item_indices][['Name', 'EpisodeCount', 'Genre', 'ImageURL']]

    return recommended_item_details


#Routes
@app.route('/')
def index():
    top_animes = pd.read_csv('./models/top_anime.csv')

    top_animes = top_animes[['Name', 'EpisodeCount', 'Genre', 'ImageURL', 'Rating']].sample(8)

    anime_images = top_animes['ImageURL'].tolist()
    
    return render_template('index.html', top_animes=top_animes, anime_images=anime_images)

@app.route('/main')
def main():
    return render_template('main.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['loginMail']
        password = request.form['loginPw']

        login_new = Login(email=email,password=password)
        db.session.add(login_new)
        db.session.commit()

        # Fetching top animes after signup
        top_animes = pd.read_csv('./models/top_anime.csv')
        top_animes = top_animes[['Name', 'EpisodeCount', 'Genre', 'ImageURL', 'Rating']].sample(8)
        anime_images = top_animes['ImageURL'].tolist()

        return render_template('index.html', top_animes=top_animes, anime_images=anime_images, message='Login Successful')

    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        signup = Signup(name=name, email=email, password=password)
        db.session.add(signup)
        db.session.commit()

        # Fetching top animes after signup
        top_animes = pd.read_csv('./models/top_anime.csv')
        top_animes = top_animes[['Name', 'EpisodeCount', 'Genre', 'ImageURL', 'Rating']].sample(8)
        anime_images = top_animes['ImageURL'].tolist()

        return render_template('index.html', top_animes=top_animes, anime_images=anime_images, message='Signup Successful')

    # For GET request, render the signup form
    return render_template('signup.html')

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        anime_name = request.form['animeName']
        anime_count = request.form['animeCount']
        content_based_rec = content_based_recommendations(train_data, anime_name, int(anime_count))

        if content_based_rec.empty:
            return render_template('main.html', message='Anime not found in the dataset')
        else:
            recommend_animes = content_based_rec.to_dict(orient='records')
            return render_template('main .html', recommend_animes=recommend_animes)


    return render_template('recommend.html')



if __name__ == '__main__':
    app.run(debug=True)