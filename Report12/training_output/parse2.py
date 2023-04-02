train_list = list()
valid_list = list()

with open('run_3.txt', 'r') as f:
    for line in f:
        if 'LOSS train' in line:
            line = line.split()
            print(line)
            train_list.append(float(line[2]))
            valid_list.append(float(line[4]))
        
print(train_list)
print(valid_list)