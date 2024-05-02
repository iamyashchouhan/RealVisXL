from flask import Flask, request, jsonify
from gradio_client import Client

app = Flask(__name__)

@app.route('/', methods=['GET'])
def get_prediction():

    # Extract data for prediction
    prompt = request.args.get("prompt")
    negative_prompt = request.args.get("negative_prompt")
    width = int(request.args.get("width"))
    height = int(request.args.get("height"))

    # Make prediction using Gradio Client
    client = Client("https://ddosxd-realvisxl.hf.space/--replicas/flm7z/")
    result = client.predict(
        prompt,
        negative_prompt,
        True,
        0,
        width,
        height,
        7,
        True,
        api_name="/run"
    )

    # Return the result as JSON response
    return jsonify(result)
