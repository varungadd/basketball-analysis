# basketball/init_db.py
from . import app, db
from .models import User, Video

with app.app_context():
    db.create_all()
    print("Database tables created successfully.")
