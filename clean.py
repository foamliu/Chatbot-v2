import os

if __name__ == '__main__':
    folder = 'data/douban-multiturn-100w'
    files = ['train.txt', 'dev.txt', 'test.txt']

    for file in files:
        filename = os.path.join(folder, file)

        with open(filename, 'r', encoding='utf-8') as f:
            data = f.readlines()

        lines = []
        for item in data:
            line = item.replace(' ', '')
            line = line + '\n'
            lines.append(line)

        with open(file, 'w', encoding='utf-8') as f:
            f.writelines(lines)
