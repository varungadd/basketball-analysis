# basketball/routes.py
import os
from flask import render_template, redirect, url_for, request, flash
from flask_login import login_user, current_user, logout_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime
from . import app, db, login_manager  # Import app and db from __init__.py

from .models import User, Video
from .video_analysis import analyze_video  # Import the analyze_video function

# Define the user_loader function
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('user_dashboard' if current_user.role == 'user' else 'admin_dashboard'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('user_dashboard' if user.role == 'user' else 'admin_dashboard'))
        flash('Invalid username or password', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = generate_password_hash(request.form['password'], method='pbkdf2:sha256', salt_length=8)
        role = request.form['role']
        new_user = User(username=username, email=email, password=password, role=role)
        db.session.add(new_user)
        db.session.commit()
        flash('Account created successfully', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/user_dashboard')
@login_required
def user_dashboard():
    user_videos = Video.query.filter_by(user_id=current_user.id).all()
    return render_template('user_dashboard.html', user=current_user, videos=user_videos)

@app.route('/admin_dashboard')
@login_required
def admin_dashboard():
    if current_user.role != 'admin':
        return redirect(url_for('user_dashboard'))

    # Fetch all videos grouped by user
    from collections import defaultdict
    players = defaultdict(list)
    all_videos = Video.query.all()

    for video in all_videos:
        players[video.user].append(video)

    # Pass the players dictionary to the template
    return render_template('admin_dashboard.html', user=current_user, players=players)


@app.route('/player_detail/<int:player_id>')
@login_required
def player_detail(player_id):
    # Fetch player details
    player = User.query.get_or_404(player_id)
    # Fetch all videos uploaded by the player
    videos = Video.query.filter_by(user_id=player_id).all()

    return render_template('player_detail.html', player=player, videos=videos)
@app.route('/upload_video', methods=['POST'])
@login_required
def upload_video():
    if 'file' not in request.files:
        flash('No file part', 'danger')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(request.url)

    # Secure the filename
    filename = secure_filename(file.filename)
    # Create a unique filename with a timestamp
    timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    unique_filename = f"{current_user.username}_{timestamp}_{filename}"

    # Define the upload folder
    upload_folder = os.path.join(app.root_path, 'static', 'uploads')
    # Create the folder if it doesn't exist
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    # Save the file to the upload folder
    file_path = os.path.join(upload_folder, unique_filename)
    file.save(file_path)

    # Create a new video record in the database
    video_record = Video(user_id=current_user.id, filename=unique_filename)
    db.session.add(video_record)
    db.session.commit()

    try:
        # Analyze the video
        analyze_video(video_record)
        flash('Video uploaded and analyzed successfully.', 'success')
    except Exception as e:
        flash(f'Video analysis failed: {e}', 'danger')

    # Redirect back to the user dashboard
    return redirect(url_for('user_dashboard'))

@app.route('/view_statistics')
@login_required
def view_statistics():
    # Your logic for viewing statistics goes here
    return render_template('view_statistics.html')


@app.route('/games')
def games():
    return render_template('games.html')

@app.route('/teams')
def teams():
    return render_template('teams.html')

@app.route('/team1')
def team1():
    return render_template('team1.html')

@app.route('/schedule')
def schedule():
    return render_template('schedule.html')

@app.route('/standings')
def standings():
    return render_template('standings.html')

@app.route('/favorite_players')
def favorite_players():
    return render_template('favorite_players.html')

@app.route('/favourite_teams')
def favourite_teams():
    return render_template('favourite_teams.html')
# routes.py
@app.route('/news')
def news():
    return render_template('news.html')

@app.route('/_navbar')
def navbar():
    return render_template('_navbar.html')
