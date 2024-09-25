from flask import Flask, request, render_template, session, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from datetime import datetime

# Reading the dataset
top_animes = pd.read_csv('./top_animeNew.csv')
train_data = pd.read_csv('./AnimeDataWithTags.csv')

# Import your database model
from models import db, User  # Import the db instance and User model

app = Flask(__name__)

# Database Configuration
app.secret_key = '72fade5da4a96210901796a122f06a9e'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/users'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy with the app context
db.init_app(app)  # This initializes the db with the app

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@app.context_processor
def inject_user():
    return dict(current_user=current_user)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def index():
    top_animes = pd.read_csv('./models/top_anime.csv')
    top_animes = top_animes[['Name', 'EpisodeCount', 'Genre', 'ImageURL', 'Rating']].sample(8)
    anime_images = top_animes['ImageURL'].tolist()

    random_animes = train_data[['Name', 'EpisodeCount', 'Genre', 'ImageURL', 'Rating']].sample(10)

    # Get the current time
    current_time = int(datetime.now().strftime('%H'))

    if 0 <= current_time < 6:
        tod = pd.read_csv('./TOD3.csv')
    elif 6 <= current_time < 12: 
        tod = pd.read_csv('./TOD1.csv')
    elif 12 <= current_time < 18:
        tod = pd.read_csv('./TOD2.csv')
    else:
        tod = pd.read_csv('./TOD4.csv')

    print(tod.head(5))

    tod = tod[['Name', 'EpisodeCount', 'Genre', 'ImageURL', 'Rating']].to_dict(orient='records')

   
    
    return render_template('index.html', top_animes=top_animes, anime_images=anime_images, random_animes=random_animes, tod=tod)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')  # Store the raw password directly
        
        new_user = User(name=name, email=email, password=password)  # No hashing here
        db.session.add(new_user)
        db.session.commit()
        
        flash('Account created successfully!', 'success')
        return redirect(url_for('login'))
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        
        if user and user.password == password:  # Direct comparison without hashing
            login_user(user)
            flash('Logged in successfully!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Login failed. Check your email and password', 'danger')
    
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('index'))  

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', user=current_user)

@app.route('/buy_item', methods=['POST'])
@login_required
def buy_item():
    item = request.form.get('item')
    if current_user.bought_items:
        current_user.bought_items += f',{item}'
    else:
        current_user.bought_items = item
    
    db.session.commit()
    flash(f'Bought item: {item}', 'success')
    return redirect(url_for('dashboard'))

# Content-based recommendation system
def content_based_recommendations(train_data, item_name, top_n):
    # Checking if the item exists in the dataset
    if item_name not in train_data['Name'].values:
        return pd.DataFrame()

    # Create a TF-IDF vectorizer for the 'Tags' column
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix_content = tfidf_vectorizer.fit_transform(train_data['Tags'])

    # Calculate cosine similarity
    cosine_similarities_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)

    # Find item index
    item_index = train_data[train_data['Name'] == item_name].index[0]

    # Get cosine similarity scores
    similar_items = list(enumerate(cosine_similarities_content[item_index]))

    # Sort items by similarity score
    similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)

    # Get top N most similar items (excluding the item itself)
    top_similar_items = similar_items[1:top_n+1]
    recommended_item_indices = [x[0] for x in top_similar_items]

    # Fetch details of recommended items
    recommended_item_details = train_data.iloc[recommended_item_indices][['Name', 'EpisodeCount', 'Genre', 'ImageURL']]

    return recommended_item_details

# Routes
@app.route('/main')
def main():
    return render_template('main.html')

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        anime_name = request.form['animeName']
        anime_count = 4
        content_based_rec = content_based_recommendations(train_data, anime_name, anime_count)

        if content_based_rec.empty:
            return render_template('main.html', message='Anime not found in the dataset')
        else:
            recommend_animes = content_based_rec.to_dict(orient='records')
            return render_template('main.html', recommend_animes=recommend_animes)

if __name__ == '__main__':
    with app.app_context():  # Create an application context
        db.create_all()  # Create tables if they don't exist
    app.run(debug=True)
