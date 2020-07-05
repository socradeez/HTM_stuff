from PIL import Image as Img
from numpy import asarray
from numpy import *
import numpy as np
from bitarray import bitarray
import cv2
import math
from htm import encoders
import random
import tkinter as tk
from tensorflow.keras.datasets import mnist
import cProfile


class Encoding_Layer():

	def __init__(self, min_val, max_val, buckets, active_bits):
		self.min_val = min_val
		self.max_val = max_val
		self.buckets = buckets
		self.active_bits = active_bits
		self.total_bits = buckets + active_bits - 1
		self.range = max_val - min_val
		self.bitarray = bitarray(self.total_bits)


	def encode_value(self, value): 
		self.bitarray.setall(False) #set the bitarray to false, otherwise it initializes with some random value I haven't figured out
		self.i = math.floor(self.buckets * (value - self.min_val) / self.range) #function to get bucket i
		indexmap = list(range(len(self.bitarray))) 
		random.seed(4)
		random.shuffle(indexmap) #indexmap creates a new list of indices of bits, then shuffles it around at random to create an index map to randomize the distribution
		for x in range(encoder.active_bits):
			self.bitarray[indexmap[encoder.i + x]] = True #iterate through the # of active bits to turn on each one via the index map created above
		return(self.bitarray)

class Neuron():
	def __init__(self, parent, index):
		self.parent = parent
		self.index = index
		self.synapses = Synapse_Map(self, self.parent.input_size)
		self.state = False
		self.boost = 1

class Minicolumn():
	def __init__(self, parent, index, depth):
		self.parent = parent
		self.index = index
		self.depth = depth
		self.boost = 1
		self.synapses = Synapse_Map(self, self.parent.input_size) #Axonal synapse mappings
		self.minlocalactivity = 0
		self.neighbors = []
		self.state = False

class Synapse_Map():
	def __init__(self, parent, length):
		self.connectedPerm = 0.2
		self.parent = parent
		self.is_trunc = False
		self.length = length
		if self.length > 10000:
			self.length = 10000
			self.is_trunc = True
		self.perms = np.random.rand(self.length)
		self.connected = self.perms > self.connectedPerm
		self.connected = bitarray(list(self.connected))

	def get_overlap(self):
		self.input_map = np.unpackbits(np.array(self.parent.parent.input))
		if self.is_trunc:
			np.random.seed(self.parent.index)
			self.input_map = np.random.choice(self.input_map, 10000)
		self.active = self.connected & bitarray(list(self.input_map))
		self.overlap = self.active.count(True) * self.parent.boost
		return(self.overlap)

	def learn(self):
		perm_delta = np.zeros(len(self.perms))
		for i in range(len(self.perms)):
			if self.active[i] == True:
				perm_delta[i] = 0.03
			else:
				perm_delta[i] = -0.015
		self.perms = np.add(perm_delta, self.perms)
		self.perms = np.clip(self.perms, 0, 1)

	def learn_training(self):
		self.input_map = self.parent.parent.input
		if self.is_trunc:
			np.random.seed(self.parent.index)
			self.input_map = bitarray(list(np.random.choice(self.input_map, 10000)))
		self.active = self.connected & self.input_map
		for i in range(len(self.perms)):
			if self.active[i] == True:
				self.perms[i] += 0.03
				self.perms[i] = min(1.0, self.perms[i])
			else:
				self.perms[i] += 0.015
				self.perms[i] = max(0.0, self.perms[i])

class TM_Layer():
	def __init__(self, input_size, num_minicolumns, depth):
		self.currentStep = 0
		self.input_size = input_size
		self.minicolumns = []
		index = 0
		for x in range(num_minicolumns):
			self.minicolumns.append(Minicolumn(self, x, depth))
			index = index + 1
		self.columnsperinharea = 30
		self.averageRecFieldSize = 10000

	def new_step(self, input_vector):
		self.input = input_vector
		self.feed_forward()

	def feed_forward(self):
		self.output_array = bitarray(0)
		for mc in self.minicolumns:
			mc.synapses.get_overlap()
		for mc in self.minicolumns:
			num_neighbors = math.floor((self.averageRecFieldSize / self.input_size) * len(self.minicolumns))
			mc.neighbors = random.sample(mc.parent.minicolumns, num_neighbors)
			neighbor_overlaps = []
			for neighbor in mc.neighbors:
				neighbor_overlaps.append(neighbor.synapses.overlap)
			neighbor_overlaps.sort(reverse=True)
			mc.minlocalactivity = neighbor_overlaps[self.columnsperinharea]
			if mc.synapses.overlap > mc.minlocalactivity:
				mc.state = True
			self.output_array.append(mc.state)

	def learn(self):
		totalfieldsize = 0
		for mc in self.minicolumns:
			if mc.state == True:
				mc.synapses.learn()
			totalfieldsize += mc.synapses.connected.count(True)
		self.averageRecFieldSize = math.floor(totalfieldsize / len(self.minicolumns))

class Classification_Layer():
	def __init__(self, input_vector_size, size):
		self.input_size = input_vector_size
		self.size = size
		self.neurons = []
		for x in range(size):
			self.neurons.append(Neuron(self, x))
		self.bitarray = bitarray(10)
		self.winner = 0

	def new_step(self, input_array):
		self.input = input_array
		self.train()


	def set_active(self, answer):
		self.bitarray.setall(False)
		self.bitarray[answer - 1] = True
		for neuron in self.neurons:
			if neuron.index == answer - 1:
				neuron.state = True
			else:
				neuron.state = False

	def train(self):
		for neuron in self.neurons:
			if neuron.state == True:
				neuron.synapses.learn_training()

	def test(self, input_array):
		self.input = input_array
		self.bitarray.setall(False)
		self.winner = 0
		highest_overlap = 0
		for neuron in self.neurons:
			neuron.synapses.get_overlap()
		for neuron in self.neurons:
			if neuron.synapses.overlap > highest_overlap:
				highest_overlap = neuron.synapses.overlap
				self.winner = neuron.index
		print("I think it's a " + str(self.winner))

def IA_to_bitarray(IA, encoder):
	IA = IA.flatten()
	encoded_output = bitarray(0)
	for x in IA:
		encoded_output.extend(encoder.encode_value(x))
	IA = encoded_output
	return(IA)

(IA_train, label_train), (IA_test, label_test) = mnist.load_data()

IA_train = IA_train[:500]
label_train = label_train[:500]
IA_test = IA_test[:200]
label_test = label_test[:200]

encoder = Encoding_Layer(0, 256, 256, 12)
print('creating TM Layer')
myAI = TM_Layer(209328, 2048, 5)
print('TM Layer Created')
classifier = Classification_Layer(2048, 10)
index = 0

for IA in IA_train:
	IA = IA_to_bitarray(IA, encoder)
	myAI.new_step(IA)
	classifier.set_active(label_train[index])
	classifier.new_step(myAI.output_array)
	print('finished ' + str(index + 1))
	myAI.learn()
	index = index + 1

print('Completed training!')
print('Onto Testing!')
index = 0
totalguesses = 0
wrong_guess = 0
for IA in IA_test:
	IA = IA_to_bitarray(IA, encoder)
	myAI.new_step(IA)
	classifier.test(myAI.output_array)
	print('The actual answer was ' + str(label_test[index]))
	if label_test[index] != classifier.winner:
		wrong_guess += 1
	myAI.learn()
	index = index + 1
	totalguesses = totalguesses + 1
	print('current error is ' + str(wrong_guess / totalguesses))


'''window = tk.Tk()
window.title("The birth of AGI")
squares_and_lines = tk.Frame(window).pack()
fv_width = len(encoder.bitarray) * 10
print(fv_width)
fv_height = 500
full_view = tk.Canvas(window, width=fv_width, height=fv_height)
start_x = 0
start_y = 0
index = 0
for bit in encoder.bitarray:
	if bit == True:
		fill_color = 'blue'
	else:
		fill_color = 'red'

	full_view.create_rectangle(start_x, start_y, start_x + 10, start_y + 10, fill=fill_color)
	start_x = start_x + 10
full_view.pack()
window.mainloop()'''

