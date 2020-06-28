# import numpy as np
# train_labels = np.array([0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0])
# labels_set = [0,1,2,3,4,5,6,7,8,9]


# label_to_indices = {label: np.where(train_labels == label)[0]
#   for label in labels_set}

# # print(label_to_indices)


# # print(np.where(train_labels == 1))


# positive_index = np.random.choice(label_to_indices[0])

# print(positive_index)

import xml.etree.ElementTree as ET
tree = ET.parse('/Users/pranoyr/Downloads/Veri/train_label.xml', parser=ET.XMLParser(encoding='iso-8859-5'))
root = tree.getroot()

images_list = []
labels_list = []
for item in root.findall("Items"):
	for item1 in item.findall("Item"):
		attrib = item1.attrib
		print(attrib)
		break
		images_list.append(attrib['imageName'])
		labels_list.append(int(attrib['vehicleID']))







