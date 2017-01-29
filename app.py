from flask import Flask, render_template, request, send_from_directory, session
from threading import Thread
from network import Network
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'

networks = {}

@app.route('/')
@app.route('/index')
@app.route('/index.html')
@app.route('/index.htm')
@app.route('/index.php')
def index():
	return render_template("index.html")


@app.route('/runnetwork', methods=["POST"])
def runnetwork():
	content = request.get_json()
	if not content:
		return "No JSON posted"

	networks[content["room"]] = Network(content)

	t = Thread(target=networks[content["room"]].process_network)
	t.start()

	return "JSON posted successfully"


@app.route('/results')
def results():
	return render_template("results.html")


@app.route('/status')
def status():
	room = request.args.get("room")
	return json.dumps(networks[room].get_state())

if __name__ == "__main__":
    app.run(debug=True)
