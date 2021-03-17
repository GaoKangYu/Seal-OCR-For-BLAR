with open('characters_top_170.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    with open('characters_top.txt', 'a', encoding='utf-8') as f2:
        for line in lines:
            word = line.split(' ')[1]
            f2.writelines(word)
