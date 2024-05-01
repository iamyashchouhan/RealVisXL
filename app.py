from flask import Flask, render_template, request, jsonify, send_from_directory
from flask import Flask, request, jsonify
from gradio_client import Client
from flask_cors import CORS


app = Flask(__name__)

@app.route("/mbsa")
def mbsa():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    # Get data from the request
    data = request.json

    # Extract parameters from the request data
    prompt = data.get("prompt")
    negative_prompt = data.get("negative_prompt")
    use_negative_prompt = data.get("use_negative_prompt")
    seed = data.get("seed")
    width = data.get("width")
    height = data.get("height")
    guidance_scale = data.get("guidance_scale")
    randomize_seed = data.get("randomize_seed")

    # Make prediction using Gradio Client
    client = Client("https://ddosxd-realvisxl.hf.space/--replicas/flm7z/")
    result = client.predict(
        prompt,
        negative_prompt,
        use_negative_prompt,
        seed,
        width,
        height,
        guidance_scale,
        randomize_seed,
        api_name="/run"
    )

    # Return the result as JSON response
    return jsonify(result)

if __name__ == '__main__':
    app.run(port=8080, debug=True)


