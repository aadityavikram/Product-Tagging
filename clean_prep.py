import json
import ast
import requests
import os
from PIL import Image

output_path = "dataset\\train"
if os.path.exists(output_path)==False:
	os.system("mkdir " + output_path)
if os.path.exists("dataset\\test")==False:
	os.system("mkdir " + "dataset\\test")

products_raw = json.load(open('productsDataset.json'))				#loading the dataset
l=len(products_raw)
for i in range(0,l):
	c1=0
	a=products_raw[i]
	cur_path = output_path + "\\" + a["tags"] + "\\"
	b1=ast.literal_eval(a['images'])  								#converting string to dictionary
	if "1" in b1.keys():              								#checking if product has any image
		c1=1
	b=a['tags']
	countunder=countspace=0           								#checking the tag format. Eg-> OUTERWEAR_JACKETS
	for j in range(0,len(b)):
		if b[j]=='_':
			countunder+=1
		elif b[j]==' ':
			countspace+=1
	if countunder==1 and countspace==0 and c1==1:					#if tags are in correct format and product has an image
		if os.path.exists(output_path + '\\' + a["tags"])==False:
			os.system("mkdir " + output_path + '\\' + a["tags"])	#creating directories for different categories/labels
		curr_path = output_path + "\\" + a["tags"] + "\\"
		print("Downloading image %s"%i)								#downloading product images
		image_url = b1["1"]
		r = requests.get(image_url)
		with open(curr_path+ "%s.jpg"%i,'wb') as f2: 
	  		f2.write(r.content)
		try:
			Image.open(curr_path+ "%s.jpg"%i).verify()
			print("Image "+str(i)+" saved in "+curr_path)			#if the downloaded image is not corrupt then saving it to its category folder
		except Exception:
			os.remove(curr_path+ "%s.jpg"%i)						#removing corrupt images