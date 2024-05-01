from flask import Flask, render_template, request, jsonify
from gradio_client import Client

app = Flask(__name__)

@app.route("/mbsa")
def mbsa():
    return render_template('index.html')

@app.route('/get_response', methods=['GET'])
def get_response():
    # Get data from the request
    prompt = request.args.get("prompt")
    negative_prompt = request.args.get("negative_prompt")
    use_negative_prompt = request.args.get("use_negative_prompt")
    seed = request.args.get("seed")
    width = request.args.get("width")
    height = request.args.get("height")
    guidance_scale = request.args.get("guidance_scale")
    randomize_seed = request.args.get("randomize_seed")

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
