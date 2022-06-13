from flask import Flask
import os

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = os.path.join(script_dir, "uploads")
app.config['DOWNLOAD_FOLDER'] = os.path.join(script_dir, "downloads")
app.config['DATA_FOLDER'] = os.path.join(script_dir, "data")
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024