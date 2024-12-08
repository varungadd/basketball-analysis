# basketball/app.py
from . import app, db
from .models import User, Video

# Create the database tables if they do not exist
with app.app_context():
    db.create_all()  # This line ensures that the tables are created automatically

if __name__ == '__main__':
    app.run(debug=True)
