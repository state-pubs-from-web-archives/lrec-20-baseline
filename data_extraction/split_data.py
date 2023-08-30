file = open("./all_tsv.tsv","r")

data = file.read().strip().split("\n")
pos,neg = [], []
for item in data:
	item = item.split("\t")
	label = item[2]
	if '1' in label:
		pos.append(item)
	else:
		neg.append(item)
print(len(pos))
print(len(neg))
import random
random.seed(0)
def class_split(data_list):
	train,val,test = [], [], []
	total_length = len(data_list)
	test_ratio = 0.1
	test_length = int(total_length * 0.1)
	test = random.sample(data_list,test_length)
	for item in data_list:
		if item not in test:
			train.append(item)
	val_length = 0.1
	total_length = len(train)
	val = random.sample(train,int(total_length*0.1))
	train_withoutval = []
	for item in train:
		if item not in val:
			train_withoutval.append(item)
	return train, val, test

pos_train, pos_val, pos_test = class_split(pos)
neg_train, neg_val, neg_test = class_split(neg)

print(len(pos_train))
print(len(pos_val))
print(len(pos_test))
print(len(neg_train))
print(len(neg_val))
print(len(neg_test))

train_file = open("./train.tsv","w")
val_file = open("./val.tsv","w")
test_file = open("./test.tsv","w")

train = pos_train + neg_train
val = pos_val + neg_val
test = pos_test + neg_test
for item in train:
	train_file.write("\t".join(item) + "\n")
for item in val:
	val_file.write("\t".join(item)+'\n')
for item in test:
	test_file.write("\t".join(item)+'\n')


