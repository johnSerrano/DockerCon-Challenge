from flask import Flask, render_template, request, send_from_directory, session
from threading import Thread
from network import process_network
from flask_socketio import SocketIO, emit, send, join_room, rooms, disconnect

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode="threading")


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
	print content["layers"]
	print content["dataset"]
	print content["iterations"]
	print content["room"]
	t = Thread(target=process_network, args=(content, progress_socket_callback))
	t.start()
	return "JSON posted successfully"
	return process_network(content, progress_socket_callback)


@app.route('/results')
def results():
	return render_template("results.html")


@app.route('/test_post.html')
def test_post():
	return render_template("test_post.html")


#use websockets to post the progress and results of the network
def progress_socket_callback(progress, room):
	socketio.emit('progress', progress, room=room)
	if progress["done"]:
		print "Done!",
	print str(progress["current_epoch"]) + '/' + str(progress["total_epochs"])
	print str(progress["val_acc"])


@socketio.on('connect')
def test_connect():
    print('socketio connected')


@socketio.on('join')
def join(message):
    join_room(message['room'])
    session['receive_count'] = session.get('receive_count', 0) + 1
    socketio.emit("join ack", room=message['room'])


@socketio.on('disconnect request')
def disconnect_request():
	session['receive_count'] = session.get('receive_count', 0) + 1
	print "****disconnected****"
	disconnect()


if __name__ == "__main__":
    socketio.run(app, debug=True)
