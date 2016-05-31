from flask import Flask, render_template, request, send_from_directory
from threading import Thread
from network import process_network
from flask_socketio import SocketIO, emit, send

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)


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
	t = Thread(target=process_network, args=(content, progress_socket_callback))
	t.start()
	return "JSON posted successfully"
	return process_network(content, progress_socket_callback)

@app.route('/results')
def results():
	return render_template("runnetwork.html")

@app.route('/test_post.html')
def test_post():
	return render_template("test_post.html")

@app.route('/social.css')
def css():
	return render_template("social.css")


#use websockets to post the progress and results of the network
def progress_socket_callback(progress):
	# send(progress)
	if progress["done"]:
		print "Done!",
	print str(progress["current_epoch"]) + '/' + str(progress["total_epochs"])
	print str(progress["val_acc"])

@socketio.on('connect', namespace='/test')
def test_connect():
    print('socketio connected')

if __name__ == "__main__":
    socketio.run(app, debug=True)
