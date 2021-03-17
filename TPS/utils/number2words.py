# map:['a', 'b' ...]
def getCharMap(file='./characters_top_170.txt'):
    dist = []
    with open(file, 'r', encoding='utf-8') as f:
        words = f.readlines()
        for word in words:
            dist.append(word.split(' ')[1].split('\n')[0])
    return dist


def convert(fileList = ['./train.txt']):
    map = getCharMap()
    for file in fileList:
        label = []
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            word = ''
            for line in lines:
                fileName = line.split(' ')[0]
                indexes = line.split(' ')[1:]
                indexes[-1] = indexes[-1].split('\n')[0]
                for index in indexes:
                    word += map[int(index)]
                label.append('img/' + fileName + '\t' + word + '\n')
                word = ''
        with open(file, 'w', encoding='utf-8') as f1:
            f1.writelines(label)

convert()