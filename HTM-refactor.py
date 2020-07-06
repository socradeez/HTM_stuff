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

class RDSE_integer():
	def __init__(self, min_val, max_val, buckets, active_bits):
		self.min_val = min_val
		self.max_val = max_val
		self.buckets = buckets
		self.seed = random.randint(1, 1000)
		self.active_bits = active_bits
		self.total_bits = buckets + active_bits - 1
		self.range = max_val - min_val
		self.bitarray = np.zeros(self.total_bits, dtype=bool)

	def encode_value(self, value): 
		self.bitarray = np.zeros(self.total_bits, dtype=bool)
		self.i = math.floor(self.buckets * (value - self.min_val) / self.range) #function to get bucket i
		indexmap = list(range(len(self.bitarray))) 
		random.seed(self.seed)
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

class Minicolumn(Neuron):
	def __init__(self, parent, index, depth):
		Neuron.__init__(self, parent, index)
		self.depth = depth
		self.minlocalactivity = 0
		self.neighbors = []

class Synapse_Map():
	def __init__(self, parent, length):
		self.connectedPerm = 0.6
		self.parent = parent
		self.length = length
		self.is_trunc = False
		if self.length > 10000:
			self.length = 10000
			self.is_trunc = True
		self.perms = np.random.rand(self.length)
		self.connected = self.perms > self.connectedPerm
		


	def get_overlap(self):
		self.input_map = self.parent.parent.input
		if self.is_trunc:
			np.random.seed(self.parent.index)
			self.input_map = np.random.choice(self.input_map, 10000)
		self.active = np.bitwise_and(self.connected, self.input_map)
		self.overlap = np.count_nonzero(self.active) * self.parent.boost
		return(self.overlap)

	def learn(self):
		self.input_map = self.parent.parent.input
		if self.is_trunc:
			np.random.seed(self.parent.index)
			self.input_map = np.random.choice(self.input_map, 10000)
		perm_delta = np.zeros(len(self.perms))
		self.active = np.bitwise_and(self.connected, self.input_map)
		for i in range(len(self.perms)):
			if self.active[i] == True:
				perm_delta[i] = 0.03
			else:
				perm_delta[i] = -0.015

		self.perms = np.add(perm_delta, self.perms)
		self.perms = np.clip(self.perms, 0, 1)

class Layer():
	def __init__(self, input_vector_size, size):
		self.input_size = input_vector_size
		self.size = size

	def new_step(self, input_vector):
		self.input = input_vector
		self.proc_input()

class SP_Layer(Layer):
	def __init__(self, input_size, size):
		Layer.__init__(self, input_size, size)
		self.minicolumns = []
		for x in range(self.size):
			self.minicolumns.append(Minicolumn(self, x, 1))
		self.columnsperinharea = 25
		self.averageRecFieldSize = input_size

	def proc_input(self):
		self.output_array = np.zeros(self.size, dtype=bool)
		i = 0
		for mc in self.minicolumns:
			mc.synapses.get_overlap()
		for mc in self.minicolumns:
			num_neighbors = math.floor((self.averageRecFieldSize / self.input_size) * len(self.minicolumns))
			random.seed(mc.index)
			mc.neighbors = random.shuffle(mc.parent.minicolumns)
			mc.neighbors = mc.parent.minicolumns[:num_neighbors]
			neighbor_overlaps = []
			for neighbor in mc.neighbors:
				neighbor_overlaps.append(neighbor.synapses.overlap)
			neighbor_overlaps.sort(reverse=True)
			mc.minlocalactivity = neighbor_overlaps[self.columnsperinharea]
			if mc.synapses.overlap > mc.minlocalactivity:
				mc.state = True
				self.output_array[i] = True
			i = i + 1

	def learn(self):
		totalfieldsize = 0
		for mc in self.minicolumns:
			if mc.state == True:
				mc.synapses.learn()
			totalfieldsize += np.count_nonzero(mc.synapses.connected)
		self.averageRecFieldSize = math.floor(totalfieldsize / len(self.minicolumns))

class Classification_Layer(Layer):
	def __init__(self, input_size, size):
		Layer.__init__(self, input_size, size)
		self.neurons = []
		for x in range(size):
			self.neurons.append(Neuron(self, x))
		self.bitarray = np.zeros(self.size, dtype=bool)

	def set_active(self, answer):
		self.bitarray = np.zeros(self.size, dtype=bool)
		self.bitarray[answer] = True
		for neuron in self.neurons:
			if neuron.index == answer - 1:
				neuron.state = True
			else:
				neuron.state = False

	def proc_input(self):
		for neuron in self.neurons:
			if neuron.state == True:
				neuron.synapses.learn()

	def test(self, input_array):
		self.input = input_array
		self.bitarray = np.zeros(self.size, dtype=bool)
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
	encoded_output = np.empty(1, dtype=bool)
	for x in IA:
		encoded_output = np.append(encoded_output, encoder.encode_value(x), 0)
	IA = encoded_output
	return(IA)

(IA_train, label_train), (IA_test, label_test) = mnist.load_data()

IA_train = IA_train[:10]
label_train = label_train[:10]
IA_test = IA_test[:200]
label_test = label_test[:200]

encoder = RDSE_integer(0, 256, 256, 5)
myAI = SP_Layer(203841, 5000)
classifier = Classification_Layer(5000, 10)
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
	classifier.set_active(label_test[index])
	classifier.new_step(myAI.output_array)
	myAI.learn()

	index = index + 1
	totalguesses = totalguesses + 1
	print('current error is ' + str(wrong_guess / totalguesses))