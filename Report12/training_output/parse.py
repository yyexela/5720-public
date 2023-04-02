import json

train_list = list()
train_count = 60

valid_list = list()
valid_count = 60

with open('run2_2.txt', 'r') as f:
    for line in f:
        if '\"train_loss\":' in line:
            json_str = line.split(' - ')[1].strip()
            json_dict = json.loads(json_str)
            if int(json_dict['epoch']) != train_count:
                raise Exception(f"Train loss: encountered epoch {int(json_dict('epoch'))} but expected epoch {train_count}")
            else:
                train_count += 1
            train_list.append(float(json_dict['train_loss']))

        if '\"valid_loss\":' in line:
            json_str = line.split(' - ')[1].strip()
            json_dict = json.loads(json_str)
            if int(json_dict['epoch']) != valid_count:
                raise Exception(f"Valid loss: encountered epoch {int(json_dict('epoch'))} but expected epoch {valid_count}")
            else:
                valid_count += 1
            valid_list.append(float(json_dict['valid_loss']))

print(train_list)
print(valid_list)