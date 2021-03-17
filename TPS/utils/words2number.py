import os
import shutil


#传入的序号与文档序号一致，返回序号对应的那个字
def find(a):
    f = open('characters_top_170.txt',encoding='utf-8')
    lines = f.readlines()
    for line in lines:
        line_word = line.split('\n')[0].split(' ')[1]
        if (line_word == a):
            index = lines.index(line)
            return index

#基本逻辑ok，改成批处理
if __name__ == '__main__':
    label_file_path = './list.txt'
    output_test = './test-test.txt'
    f = open(label_file_path, encoding='utf-8')
    lines = f.readlines()
    for line in lines:
        x = line.split(' ')
        y = line.split(' ')[0]
        f_out = open(output_test, 'a',encoding = 'utf-8')
        f_out.writelines(y)
        length = len(x)-1
        for i in range(length):
            a = x[i+1]
            a = a.split('\n')[0]
            print('正在变换，当前字符为' + a)
            b = find(a)
            print('变换后对象为：',b)
            print(b)
            f_out.writelines(' ')
            b = str(b)
            f_out.writelines(b)
        f_out.writelines('\n')
    print('done')
 #python words2number.py