import random


if __name__ == '__main__':
    # To generate random Datasets
    dataset_size = 1000
    increment = 500

    while(dataset_size <= 50000):

        # for i in range(2):
        dataset_file_path = 'dataset/dataset_%d.txt' % (dataset_size)
        with open(dataset_file_path, 'w+', encoding="utf-8") as dataset_file:
            end = dataset_size * 0.5
            start = end * 0.1
            for j in range(dataset_size):
                x = random.randint(start, end)
                y = random.randint(start, end)

                dataset_file.write('%d %d\n' % (x, y))

        dataset_size += increment
