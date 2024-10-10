from flask import Flask, request, render_template, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
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

# Content-based recommendation system
def content_based_recommendations(train_data, user_input, top_n):
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')

    # Fit the vectorizer on the 'Tags' column from the dataset
    tfidf_matrix_content = vectorizer.fit_transform(train_data['Tags'])

    # Transform the user input (query) into a TF-IDF vector
    user_input_tfidf = vectorizer.transform([user_input])

    # Calculate cosine similarity between the user's input and all items in the dataset
    cosine_similarities_content = cosine_similarity(user_input_tfidf, tfidf_matrix_content).flatten()

    # Sort items by similarity score in descending order
    similar_items = sorted(list(enumerate(cosine_similarities_content)), key=lambda x: x[1], reverse=True)

    # Get top N most similar items
    top_similar_items = similar_items[:top_n]

    # Get the indices of recommended items
    recommended_item_indices = [x[0] for x in top_similar_items]

    # Fetch details of the recommended items
    recommended_item_details = train_data.iloc[recommended_item_indices][['Name', 'EpisodeCount', 'Genre', 'ImageURL']]

    return recommended_item_details

# Bought items
def bought_items_details(train_data, bought_items):
    if bought_items:
        bought_items = bought_items.split(',')
        bought_items_indices = train_data[train_data['Name'].isin(bought_items)].index
        bought_items_details = train_data.iloc[bought_items_indices][['Name', 'EpisodeCount', 'Genre', 'ImageURL', 'Rating']]
        # Convert the DataFrame to a list of dictionaries
        bought_items_details = bought_items_details.to_dict('records')
    return bought_items_details

# Collaborative filtering recommendation system
def collborative_filtering_recommendations(train_data, target_user_id, top_n=10):
    #Create the user_item_matrix 
    user_item_matrix=train_data.pivot_table(index='UserID', columns='AnimeID', values='Rating', aggfunc='mean').fillna(0).astype(int)

    #Calculate the user similarity matrix using cosine similarity
    user_similarity = cosine_similarity(user_item_matrix)

    #Find the index of the target user in the matrix
    target_user_index = user_item_matrix.index.get_loc(target_user_id)

    #Get the similarity scores for the target user
    user_similarities = user_similarity[target_user_index]

    #Sort the users by similarity in descending order
    similar_users_indices = user_similarities.argsort()[::-1][1:]

    #Generate recommendation based on similarities
    recommended_items = []

    for user_index in similar_users_indices:
        #Get items rated by the similar users
        rated_by_similar_user = user_item_matrix.iloc[user_index]
        not_rated_by_target_user = (rated_by_similar_user == 0) & (user_item_matrix.iloc[target_user_index])

        #Extract the item IDs
        recommended_items.extend(user_item_matrix.columns[not_rated_by_target_user][:top_n])

    #Get the details of recommended items
    recommended_items_details = train_data[train_data['AnimeID'].isin(recommended_items)][['Name', 'EpisodeCount', 'Genre', 'ImageURL', 'Rating']]

    recommended_items_details = recommended_items_details.to_dict('records')

    return recommended_items_details

def get_tags_for_partial_anime(train_data, user_input):
    # Find rows where the anime name contains the user input
    anime_rows = train_data[train_data['Name'].str.contains(user_input, case=False, na=False)]
    if not anime_rows.empty:
        # Return a list of tags for matched anime
        return anime_rows[['Name', 'Tags']].to_dict(orient='records')  # List of dicts with Name and Tags
    return []

def evaluate_recommendations(user_input, recommended_items, train_data):

    # Get all anime that match the partial user input
    matching_animes = get_tags_for_partial_anime(train_data, user_input)

    if not matching_animes:
        return {'precision': 0, 'recall': 0}  # No matches found

    # Create a set of relevant tags based on matching animes
    relevant_tags = set()
    for anime in matching_animes:
        relevant_tags.update(anime['Tags'].split(','))  # Add tags for each matching anime

    # Set to track true positives
    true_positives = 0

    # Check recommended items against relevant tags
    for item in recommended_items:
        item_tags = get_tags_for_partial_anime(train_data, item['Name'])
        if item_tags:
            for it in item_tags:
                item_tag_set = set(it['Tags'].split(','))
                if not relevant_tags.isdisjoint(item_tag_set):  # Check for any common tags
                    true_positives += 1
                    break  # Only count once for each recommended item

    # Precision: TP / (TP + FP)
    precision = true_positives / len(recommended_items) if recommended_items else 0

    # Recall: TP / (Relevant items)
    recall = true_positives / len(relevant_tags) if relevant_tags else 0

    return {
        'precision': precision,
        'recall': recall
    }


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
        tod = pd.read_csv('./models/TOD2.csv')
        time_of_day = 'Midnight'
    elif 6 <= current_time < 12: 
        tod = pd.read_csv('./models/TOD0.csv')
        time_of_day = 'Morning'
    elif 12 <= current_time < 18:
        print('time came here')
        tod = pd.read_csv('./models/TOD1.csv')
        time_of_day = 'Afternoon'
    elif 18 <= current_time < 24:
        tod = pd.read_csv('./models/TOD3.csv')
        time_of_day = 'Night'

    tod = tod[['Name', 'EpisodeCount', 'Genre', 'ImageURL', 'Rating']].to_dict(orient='records')
    
    return render_template('index.html', top_animes=top_animes, anime_images=anime_images, random_animes=random_animes, tod=tod, time_of_day=time_of_day)

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
    if current_user.bought_items:
        bought_items = bought_items_details(train_data, current_user.bought_items)

        user_bought_items = current_user.bought_items
        bought_items_for_collaborative = user_bought_items.split(',')
        bought_items_indices = train_data[train_data['Name'].isin(bought_items_for_collaborative)].index

        relavant_anime = train_data.loc[bought_items_indices].sample(1)  # Now sampling from DataFrame, not Index
        user_id_for_collaborative = relavant_anime['UserID'].values[0] 

        collaborative_recommendations = collborative_filtering_recommendations(train_data, user_id_for_collaborative)

        return render_template('dashboard.html', user=current_user, bought_items_details=bought_items, collaborative_animes=collaborative_recommendations)
    else:
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
            return render_template('main.html', message='Anime not found!')
        else:
            recommend_animes = content_based_rec.to_dict(orient='records')

            # Evaluate recommendations based on tags
            evaluation_results = evaluate_recommendations(anime_name, recommend_animes, train_data)

            # Render the recommendations and evaluation results
            return render_template('main.html', recommend_animes=recommend_animes, evaluation=evaluation_results)

if __name__ == '__main__':
    with app.app_context():  # Create an application context
        db.create_all()  # Create tables if they don't exist
    app.run(debug=True)
