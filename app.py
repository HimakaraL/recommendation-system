from flask import Flask, request, render_template
import pandas as pd
import random

app = Flask(__name__)

@app.route('/')
def index():
    top_animes = pd.read_csv('./models/top_anime.csv')

    top_animes = top_animes[['Name', 'EpisodeCount', 'Genre', 'ImageURL', 'Rating']].sample(8)

    anime_images = top_animes['ImageURL'].tolist()
    
    return render_template('index.html', top_animes=top_animes, anime_images=anime_images)

@app.route('/main')
def main():
    return render_template('main.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/login')
def login():
    return render_template('login.html')

if __name__ == '__main__':
    app.run(debug=True)