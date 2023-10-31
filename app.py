import os
import faiss
import spotipy
import gradio as gr
import pandas as pd
from dotenv import load_dotenv
from gradio.components import Textbox, HTML
from spotipy.oauth2 import SpotifyClientCredentials
from sentence_transformers import SentenceTransformer

load_dotenv('src/creds.env')
client_id = os.getenv('client_id')
client_secret = os.getenv('client_secret')

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))

model = SentenceTransformer('msmarco-MiniLM-L-12-v3')
music_index = faiss.read_index('data/faiss_index.bin')

tracks = pd.read_csv('data/tracks.csv', sep=';')

def get_album_cover(spotify_session, track_id):
    try:
        url = spotify_session.track(track_id)['album']['images'][0]['url']
    except:
        url = 'https://www.lifewire.com/thmb/5Y8ggTdQiyLdq9us-IMpsACJP-s=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/alert-icon-5807a14f5f9b5805c2aa679c.PNG'
    return url
def search_engine(Musica):
    query = model.encode([Musica])
    top_k = music_index.search(query, 5)
    
    found_tracks = tracks.iloc[top_k[1].tolist()[0]][['artists', 'track_name', 'track_id']]
    found_tracks['album_cover'] = found_tracks['track_id'].map(lambda x: get_album_cover(sp, x))
    
    table_template = '''
        <!DOCTYPE html>
        <html>
        <head>
        <meta charset="UTF-8">
        <title>Sua Página</title>
        <link href="https://fonts.googleapis.com/css?family=Open+Sans:400,600" rel="stylesheet">

        <style>
        *, *:before, *:after {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
        }

        body {
        background: #ffffff;
        font-family: 'Open Sans', sans-serif;
        }

        table {
        background: #ffffff;
        border-radius: 0.25em;
        border-collapse: collapse;
        margin: 1em;
        }

        th {
        border-bottom: 1px solid #000000;
        color: #000000;
        font-size: 0.85em;
        font-weight: 600;
        padding: 0.5em 1em;
        text-align: center;
        vertical-align: middle;
        }

        td {
        color: #000000;
        font-weight: 400;
        padding: 0.65em 1em;
        text-align: center;
        vertical-align: middle;
        }

        .disabled td {
        color: #959595;
        }

        tbody tr {
        transition: background 0.25s ease;
        }

        tbody tr:hover {
        background: #959595;
        }
        </style>
        </head>
        <body>
        <table align="center">
        <thead>
        <tr>
        <th>Name</th>
        <th>Artists</th>
        <th>Album Cover</th>
        </tr>
        </thead>
        <tbody>
    '''
    
    for recommendation in found_tracks.values:
        table_template += f'''
        <tr>
        <td>{recommendation[1]}</td>
        <td>{recommendation[0].replace(';', ' | ')}</td>
        <td><img src="{recommendation[3]}" width="150" height="90"></td>
        </tr>
        '''
    
    table_template += '''
        </tbody>
        </table>
        </body>
        </html>
    '''
    
    return table_template

with gr.Blocks(title='Music Search Engine') as demo:
        with gr.Box():
            gr.Markdown('## Choose Track Name or Track Artist')
            track_name = gr.Textbox(label='Semantic Search String', placeholder='Type here...', lines=1)
            with gr.Row():
                btn_track_name = gr.Button('Search')
            with gr.Row():
                btn_track_name.click(fn=search_engine, inputs=track_name, outputs=gr.HTML())

demo.launch(favicon_path='src/favicon.jpeg')