import os
from flask import Flask, redirect, url_for, session, render_template_string
from authlib.integrations.flask_client import OAuth
from dotenv import load_dotenv

from src.utils.default_config_settings import default_config
from web_ui import create_ui
from gradio import mount_gradio_app

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-me")
app.config['SESSION_COOKIE_NAME'] = 'browseruse_session'

oauth = OAuth(app)

oauth.register(
    name='google',
    client_id=os.getenv('GOOGLE_CLIENT_ID'),
    client_secret=os.getenv('GOOGLE_CLIENT_SECRET'),
    access_token_url='https://accounts.google.com/o/oauth2/token',
    access_token_params=None,
    authorize_url='https://accounts.google.com/o/oauth2/auth',
    authorize_params=None,
    api_base_url='https://www.googleapis.com/oauth2/v1/',
    userinfo_endpoint='https://www.googleapis.com/oauth2/v1/userinfo',
    client_kwargs={'scope': 'openid email profile'}
)


def login_required(func):
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        if 'google_token' not in session:
            return redirect(url_for('login'))
        return func(*args, **kwargs)
    return wrapper


@app.route('/')
def index():
    if 'google_token' in session:
        return redirect(url_for('gradio_app'))
    return render_template_string('<a href="{{ url_for("login") }}">Login with Google</a>')


@app.route('/login')
def login():
    redirect_uri = url_for('authorize', _external=True)
    return oauth.google.authorize_redirect(redirect_uri)


@app.route('/authorize')
def authorize():
    token = oauth.google.authorize_access_token()
    session['google_token'] = token
    resp = oauth.google.get('userinfo')
    session['user'] = resp.json()
    return redirect(url_for('gradio_app'))


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


demo = create_ui(default_config())
mount_gradio_app(app, demo, path="/ui")


@app.route('/app')
@login_required
def gradio_app():
    return redirect('/ui')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 7788)))
