from data_handler import Instance
import numpy as np
import matplotlib.pyplot as plt
import time
import copy


class Solution:
    def __init__(self, instance):
        if not isinstance(instance, Instance):
            raise TypeError("Error: instance variable is not data_handler.Instance. Type: " + type(instance))

        self.instance = instance
        self.radius = 0
        self.center_circle = np.zeros(2)
        self.farthest_pair = np.zeros((2, 2))

    def draw_circle(self, data, center, radius, method):
        c = plt.Circle((center[0], center[1]), radius, fill=False)
        ax= plt.gca()
        ax.add_patch(c)
        
        plt.axis("scaled")
        plt.suptitle(str(method))
        plt.title("Radius: %.4f   Center: [%.4f, %.4f]" % (radius, center[0], center[1]))
        plt.xlabel('X')
        plt.ylabel('Y')
        ax.plot(data[:,0], data[:,1], 'o', color='black', alpha=0.5)
        plt.savefig('fig/min_circle_' + str(method) + '.png')
        plt.close()

    def print_solution(self, solution_path, method, center=None, radius=None):
        filename = solution_path
        if center is not None:
            center_points = str(center[0]) + ", " + str(center[1])
            radius = str(radius)
        else:
            center_points = str(self.center_circle[0]) + ", " + str(self.center_circle[1])
            radius = str(self.radius)
        
        with open(filename, mode='w') as fp:
            fp.write(str(method) + "\n")
            fp.write("DataSet Name: " + str(self.instance.dataset_name) + "\n")
            fp.write("Number of Points: " + str(self.instance.number_points) + "\n")
            fp.write("Center of Circle: (" + center_points + ")\n")
            fp.write("Radius of Circle: " + radius + "\n")

class Heuristic:
    def __init__(self, instance, solution):
        if not isinstance(instance, Instance):
            raise TypeError("Error: instance variable is not data_handler.Instance. Type: " + type(instance))

        self.instance = instance
        self.solution = solution

    def run(self):
        self.initial_pair()
        self.construct_solution()
        self.solution.draw_circle(self.solution.instance.data_values, self.solution.center_circle, self.solution.radius, "Heuristic Algorithm")

    def initial_pair(self):
        # Find the bigest gap in x and y
        max_in_columns = np.amax(self.solution.instance.data_values, axis=0)
        min_in_columns = np.amin(self.solution.instance.data_values, axis=0)
        gap_x = max_in_columns[0] - min_in_columns[0]
        gap_y = max_in_columns[1] - min_in_columns[1]

        # Sort by x or y to find the farthest pair of points
        if gap_x >= gap_y:
            self.solution.instance.data_values = self.solution.instance.data_values[self.solution.instance.data_values[:,0].argsort()]
        else:
            self.solution.instance.data_values = self.solution.instance.data_values[self.solution.instance.data_values[:,1].argsort()]

        # Fill the farthest pair of points
        self.solution.farthest_pair[0] = self.solution.instance.data_values[0]
        self.solution.farthest_pair[1] = self.solution.instance.data_values[-1]
        
        # Fill circle center for the initial pair of points
        self.solution.center_circle = (self.solution.farthest_pair[0] + self.solution.farthest_pair[1]) / 2

        # Fill radius for the first pair of points
        norm = np.linalg.norm(self.solution.farthest_pair[0] - self.solution.farthest_pair[1])
        self.solution.radius = norm / 2

    def construct_solution(self):
        for point in self.instance.data_values:
            vector_point_to_center = point - self.solution.center_circle
            norm_vector_point_to_center = np.linalg.norm(vector_point_to_center)

            # If a point is out of the circle
            if norm_vector_point_to_center > self.solution.radius:
                # Update center
                normalized_vector = vector_point_to_center / norm_vector_point_to_center
                distance_to_add_center = ((norm_vector_point_to_center - self.solution.radius) / 2) * normalized_vector
                self.solution.center_circle = self.solution.center_circle + distance_to_add_center

                # Update radius
                self.solution.radius = (norm_vector_point_to_center + self.solution.radius) / 2


class Circle:
    def __init__(self):
        self.radius = 0
        self.center = np.zeros(2)


class Random_Circle:
    def __init__(self, instance, solution):
        if not isinstance(instance, Instance):
            raise TypeError("Error: instance variable is not data_handler.Instance. Type: " + type(instance))

        self.instance = instance
        self.solution = solution
        # Added a fake position in index 0 to make implementation of algorithm easier
        self.data = np.concatenate([np.array([[0.,0.]]), self.solution.instance.data_values])
        self.circle = [Circle() for _ in range(len(self.data))]

    def run(self):
        self.min_circle()
        self.solution.draw_circle(self.data[1:], self.circle[-1].center, self.circle[-1].radius, "Exact Algorithm")

    def min_circle(self):
        self.random_permutation()

        # Initializing algorithm with first 2 elements
        circle_index = 2
        self.initialize_circle(1, 2, circle_index)

        for i in range(3, len(self.data)):
            if self.inside_circle(i):
                self.circle[i] = copy.deepcopy(self.circle[i - 1])
            else:
                self.circle[i] = self.min_circle_with_point(i - 1, 1, i)

        print(str(self.circle[-1].center))
        print(str(self.circle[-1].radius))
        return self.circle[-1]

    def min_circle_with_point(self,index_end_for, index_first_element_circle, index_second_element_circle):
        circle_index = 1
        self.initialize_circle(index_first_element_circle, index_second_element_circle, circle_index)

        for j in range(2, index_end_for + 1):
            if self.inside_circle(j):
                self.circle[j] = copy.deepcopy(self.circle[j - 1])    
            else:
                self.circle[j] = self.min_circle_with_two_points(j - 1, j, index_second_element_circle)

        return self.circle[index_end_for]

    def min_circle_with_two_points(self,index_end_for, index_first_element_circle, index_second_element_circle):
        circle_index = 0
        self.initialize_circle(index_first_element_circle, index_second_element_circle, circle_index)

        for k in range(1, index_end_for + 1):
            if self.inside_circle(k):
                self.circle[k] = copy.deepcopy(self.circle[k - 1])
            else:
                has_obtuse_angle, two_index_points_of_obtuse_angle = self.obtuse_angle(index_first_element_circle, index_second_element_circle, k)
                if has_obtuse_angle:
                    self.initialize_circle(two_index_points_of_obtuse_angle[0], two_index_points_of_obtuse_angle[1], k)
                else:
                    center, radius = self.get_circuncenter(index_first_element_circle, index_second_element_circle, k)
                    self.circle[k].center = center
                    self.circle[k].radius = radius

        return self.circle[index_end_for]

    def initialize_circle(self, index_first_element_circle, index_second_element_circle, circle_index):
        center = self.get_center(self.data[index_first_element_circle], self.data[index_second_element_circle])
        radius = self.get_radius(self.data[index_first_element_circle], self.data[index_second_element_circle])
        self.circle[circle_index].radius = radius
        self.circle[circle_index].center = center

    def inside_circle(self, index_point):
        radius_new_point = np.linalg.norm(self.data[index_point] - self.circle[index_point - 1].center)

        if self.circle[index_point - 1].radius >= radius_new_point:
            return True

        return False

    def get_center(self, first_point, second_point):
        center = (first_point + second_point) / 2
        return center

    def get_radius(self, first_point, second_point):
        norm = np.linalg.norm(first_point - second_point)
        radius = norm / 2
        return radius

    def obtuse_angle(self, index_first_point, index_second_point, index_third_point):
        points = np.array([self.data[index_first_point], self.data[index_second_point], self.data[index_third_point]])

        distance_0_1 = np.linalg.norm(points[0] - points[1])
        distance_1_2 = np.linalg.norm(points[1] - points[2])
        distance_0_2 = np.linalg.norm(points[0] - points[2])

        array_distances = np.array([distance_0_1, distance_1_2, distance_0_2])

        max_element = array_distances.argmax()

        has_obtuse_angle = False
        two_index_points_of_obtuse_angle = None

        if max_element == 0:
            if distance_0_1 ** 2 > distance_1_2 ** 2 + distance_0_2 ** 2:
                has_obtuse_angle = True
                two_index_points_of_obtuse_angle = np.array([index_first_point, index_second_point])
        elif max_element == 1:
            if distance_1_2 ** 2 > distance_0_1 ** 2 + distance_0_2 ** 2:
                has_obtuse_angle = True
                two_index_points_of_obtuse_angle = np.array([index_second_point, index_third_point])
        elif max_element == 2:
            if distance_0_2 ** 2 > distance_0_1 ** 2 + distance_1_2 ** 2:
                has_obtuse_angle = True
                two_index_points_of_obtuse_angle = np.array([index_first_point, index_third_point])

        return has_obtuse_angle, two_index_points_of_obtuse_angle

    def get_circuncenter(self, first_point_index, second_point_index, third_point_index):
            a1 = 2 * (self.data[first_point_index] - self.data[second_point_index])
            b1 = (self.data[first_point_index] ** 2 - self.data[second_point_index] ** 2).sum()

            a2 = 2 * (self.data[first_point_index] - self.data[third_point_index])
            b2 = (self.data[first_point_index] ** 2 - self.data[third_point_index] ** 2).sum()

            center = np.linalg.solve(np.array([a1, a2]), np.array([b1, b2]))
            radius = np.linalg.norm(self.data[first_point_index] - center)

            return center, radius

    def random_permutation(self):
        for k in range(len(self.data) - 1, 2, -1):
            # Start in 1 because fake element in position 0 to make easier implementation of algorithm
            # Ends in k+1 because the highest element in np.random.randin is exclusive
            index_to_change = np.random.randint(1, k+1)
            transational_element = np.copy(self.data[k])

            self.data[k] = np.copy(self.data[index_to_change])
            self.data[index_to_change] = np.copy(transational_element)