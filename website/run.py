import sys
import os
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from transparency import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
