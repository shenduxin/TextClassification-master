import pandas as pd
import random
random.seed(0)

def processData(raw_file ,mode,concat=True):
    df = pd.read_csv(raw_file, names=['label', 'title', 'content'])
    data = []
    for index, row in df.iterrows():
        title = row['title'].strip()
        content = row['content'].strip()
        if concat:
            line = title + ' ' + content + '\t' + str(row['label'] - 1)
        else:
            line = title + '\t' + content + '\t' + str(row['label'] - 1)
        data.append(line)
    # random.shuffle(data)
    if mode == 'train':
        data_0 = []
        data_1 = []
        data_2 = []
        data_3 = []
        for d in data:
            if d.rpartition('\t')[2] == '0':
                data_0.append(d)
            elif d.rpartition('\t')[2] == '1':
                data_1.append(d)
            elif d.rpartition('\t')[2] == '2':
                data_2.append(d)
            elif d.rpartition('\t')[2] == '3':
                data_3.append(d)

        train_data_0 = data_0[0:28500]
        dev_data_0 = data_0[-1500:]
        train_data_1 = data_1[0:28500]
        dev_data_1 = data_1[-1500:]
        train_data_2 = data_2[0:28500]
        dev_data_2 = data_2[-1500:]
        train_data_3 = data_3[0:28500]
        dev_data_3 = data_3[-1500:]
    
        train_data_0.extend(train_data_1)
        train_data_0.extend(train_data_2)
        train_data_0.extend(train_data_3)
        dev_data_0.extend(dev_data_1)
        dev_data_0.extend(dev_data_2)
        dev_data_0.extend(dev_data_3)
    
        random.shuffle(train_data_0)
        random.shuffle(dev_data_0)

        if concat:
            with open('datasets/AGNews/processed/train.txt', 'w') as f:
                for line in train_data_0:
                    f.write(line + '\n')
        
            with open('datasets/AGNews/processed/dev.txt', 'w') as f:
                for line in dev_data_0:
                    f.write(line + '\n')
        else:
            with open('datasets/AGNews_SEP/processed/train.txt','w') as f:
                for line in train_data_0:
                    f.write(line + '\n')
            with open('datasets/AGNews_SEP/processed/dev.txt','w') as f:
                for line in dev_data_0:
                    f.write(line + '\n')
    else:
        if concat:            
            with open('datasets/AGNews/processed/test.txt', 'w') as f:
                for line in data:
                    f.write(line + '\n')
        else:
            with open('datasets/AGNews_SEP/processed/test.txt', 'w') as f:
                for line in data:
                    f.write(line + '\n')

def main():
    processData('datasets/AGNews/raw/train.csv', 'train',False)
    processData('datasets/AGNews/raw/test.csv', 'test',False)
    print('Done!')

if __name__ == '__main__':
    main()