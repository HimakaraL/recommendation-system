from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

# The db instance will be imported from app.py
db = SQLAlchemy()

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    bought_items = db.Column(db.Text, nullable=True)  # Store bought items as a comma-separated string
