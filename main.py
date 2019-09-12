from data_handler import Instance
from min_circle import Solution, Heuristic, Random_Circle
import argparse
from os import listdir
from os.path import join
import time
from datetime import datetime
import random
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Min-circle problem")

    parser.add_argument('-o', '--solution_dir', default='solution', help="Directory to save solutions.")
    parser.add_argument('-d', '--dataset_dir', default='dataset', help='Directory containing dataset files.')
    parser.add_argument('-f', '--dataset_file', help="Dataset file to be processed.")

    args = parser.parse_args()

    if args.dataset_file is not None:
        dataset_files = [args.dataset_file]
    else:
        dataset_files = sorted(listdir(args.dataset_dir))

    if '.DS_Store' in dataset_files:
        dataset_files.remove('.DS_Store')
    
    # To plot comparison between algorithms
    num_samples_dataset = []
    time_spent_exact = []
    time_spent_heuristic = []

    for dataset_file in dataset_files:
        print("Dataset: %s" % dataset_file)

        # Path where is the dataset
        dataset_path = join(args.dataset_dir, dataset_file)
        dataset_name = dataset_file.rstrip('.txt')

        # Path and model of solution file
        time_now = datetime.now().strftime("%Y%m%d%H%M%S")
        solution_file = str(dataset_name) + "_" + str(time_now) + ".txt"
        solution_path = join(args.solution_dir, solution_file)

        # Initialization of instance and solution
        instance = Instance(dataset_path, solution_path, dataset_name)
        solution = Solution(instance)

        # Random Algorithm
        start_time_exact = time.time()
        random_circle = Random_Circle(instance, solution)
        random_circle.run()
        total_time_exact = time.time() - start_time_exact
        time_spent_exact.append(total_time_exact)
        solution.print_solution("solution/Exact_" + solution_file, "Exact Algorithm", random_circle.circle[-1].center, random_circle.circle[-1].radius)

        # Heuristic Algorithm
        start_time_heuristic = time.time()
        heuristic = Heuristic(instance, solution)
        heuristic.run()
        total_time_heuristic = time.time() - start_time_heuristic
        time_spent_heuristic.append(total_time_heuristic)
        solution.print_solution("solution/Heuristic_" + solution_file, "Heuristic Algorithm")

        num_samples_dataset.append(instance.number_points)

    print("##########################################")

    # # To print Scatter Plot of Time Spent in Heuristic and Exact Algorithms
    # fig = plt.figure()
    # ax = fig.add_subplot()
    # ax.scatter(num_samples_dataset, time_spent_exact, alpha=0.7, c='blue', label=['Exact'])
    # ax.scatter(num_samples_dataset, time_spent_heuristic, alpha=0.7, c='red', label=['Heuristic'])
    # # ax.set_yscale('log')

    # plt.title('Algorithm Comparison')
    # plt.legend(loc=2)
    # plt.xlabel('Number of points')
    # plt.ylabel('Time spent (seconds)')
    # plt.savefig('fig/solutions_comparison.png')
