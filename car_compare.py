from losses import TripletLoss
from datasets import TripletMNIST, TripletVeriDataset
from networks import EmbeddingNet, Resnet18
from metrics import AccumulatedAccuracyMetric
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

import cv2
import torch.nn as nn
from PIL import Image
import torch.optim as optim
import os
import torch
import numpy as np
import pymongo

myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["vehicledetails"]
encodings = mydb["enc"]


class TripletNet(nn.Module):
	def __init__(self, embedding_net):
		super(TripletNet, self).__init__()
		self.embedding_net = embedding_net

	def forward(self, x1):
		output1 = self.embedding_net(x1)
		return output1


def who_is_it(encodings, database, threshold=0.5):
	"""
	Arguments:
	encodings -- encodings of faces detected 
	database -- database containing image encodings along with the name of the person on the image
	Returns:
	min_dist -- the minimum distance between image_path encoding and the encodings from the database
	identity -- string, the name prediction for the person on image_path
	"""
	vehicle_details = {}
	# db_vectors=np.array(db_vectors)
	for encoding in encodings:
		# Initialize "min_dist" to a large value, say 100 (≈1 line)
		min_dist = 100
		# Loop over the database dictionary's names and encodings.
		for data in database:
			date_time, veh_type, veh_make, veh_model, db_enc = data['datetime'], data[
				'veh_type'], data['veh_make'], data['veh_model'], data['enc']
			# Compute cosine distance between the target "encoding" and the encoding from the database. (≈ 1 line)
			dist = (torch.tensor(db_enc, dtype=torch.float32) - encoding).pow(2).sum(1)
			# If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
			if dist < min_dist:
				min_dist = dist
				vehicle_details['datetime'] = date_time
				vehicle_details['veh_type'] = veh_type
				vehicle_details['veh_make'] = veh_make
				vehicle_details['veh_model'] = veh_model
		
		if min_dist > threshold:
			vehicle_details = {}

	return vehicle_details


class CarCompare():
	def __init__(self):
		embedding_net = Resnet18()
		self.model = TripletNet(embedding_net)

		self.transform = transforms.Compose([
			transforms.Resize((150, 150)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
				0.229, 0.224, 0.225])
		])

		#checkpoint = torch.load('/home/nvidia/weights/model6.pth')
		checkpoint = torch.load(
			'/Users/pranoyr/Desktop/model6.pth', map_location='cpu')
		self.model.load_state_dict(checkpoint['model_state_dict'])
		self.model.eval()

	def get_encoding(self, img, bbox):
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
		img = Image.fromarray(img)
		img = self.transform(img)
		with torch.no_grad():
			enc = self.model(img.unsqueeze(0)).numpy()
		return enc

	# def get_encoding(self, img):
	# 	# img = Image.fromarray(img)
	# 	img = self.transform(img)
	# 	with torch.no_grad():
	# 		enc = self.model(img.unsqueeze(0)).numpy()
	# 	return enc

	def add_enc(self, img, bbox):
		enc = self.get_encoding(img, bbox)
		# img3 = cv2.imread('/Users/pranoyr/Desktop/cars/entry.png')
		# img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
		# img3 = Image.fromarray(img3)
		# enc = self.get_encoding(img3)
		encodings.insert_one({"enc": enc.tolist(), 'datetime': "a", 'veh_type':'b' , 'veh_make' : 'c', 'veh_model' : 'd' })	

	def compare(self, img, bbox):
		enc = self.get_encoding(img, bbox)
		# img3 = cv2.imread('/Users/pranoyr/Desktop/cars/exit.png')
		# img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
		# img3 = Image.fromarray(img3)
		# enc = self.get_encoding(img3)
		results = who_is_it(enc , encodings.find(), threshold=0.7)
		return results

	def delete(self, results):
		encodings.delete_one(results)
			
			
if (__name__=='__main__'):
	a = CarCompare()
	# a.add_enc()
	results = a.compare()
	print(results)
	if results:
		a.delete(results)
		print("@#4@34")
		### send results
