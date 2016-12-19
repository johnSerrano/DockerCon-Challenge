import png
import os

def png_l(arr, dataset, location):
	arr *= 255
	arr = arr.reshape(dataset["img_rows"], dataset["img_cols"])

	with open(location, 'wb') as f:
		w = png.Writer(dataset["img_rows"], dataset["img_cols"], greyscale=True)
		w.write(f, arr)

	with open(location, 'rb') as f:
		data_string = f.read().encode('base64').replace('\n', '')
	os.remove(location)
	return data_string

def png_rgb(arr, dataset, location):
	arr *= 255
	arr = arr.reshape(dataset["img_rows"]*dataset["img_cols"]*dataset["img_channels"])
	pixels = []
	image = []

	for i in range(dataset["img_rows"]*dataset["img_cols"]):
		pixel = [arr[i], arr[i+(dataset["img_rows"]*dataset["img_cols"])],
				 arr[i+(dataset["img_rows"]*dataset["img_cols"]*2)]]
		pixels += [pixel]

	for i in range(dataset["img_rows"]):
		row = []
		for j in range(dataset["img_cols"]):
			row += pixels[(dataset["img_cols"]*i)+j]
		image += [row]

	with open(location, 'wb') as f:
		w = png.Writer(dataset["img_rows"], dataset["img_cols"])
		w.write(f, image)

	with open(location, 'rb') as f:
		data_string = f.read().encode('base64').replace('\n', '')
	os.remove(location)
	return data_string
