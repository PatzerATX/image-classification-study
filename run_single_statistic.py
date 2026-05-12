from statistics import mean, stdev

from naive_bayes_test import naive_bayes_test
from perceptron_test import perceptron_test

def multiple_runs(runs, train_percentage, algorithm_test,
                  training_label_path, training_image_path,
                  test_label_path, test_image_path,
                  image_type, feature_type):

    accuracy_results = []
    elapsed_time_results = []

    for run in range(runs):
        accuracy, elapsed_time = algorithm_test(
                    training_label_path, training_image_path,
                    test_label_path, test_image_path,
                    image_type, feature_type, train_percentage)
        accuracy_results.append(accuracy)
        elapsed_time_results.append(elapsed_time)

    average_accuracy = mean(accuracy_results)
    std_accuracy = stdev(accuracy_results)
    average_elapsed_time = mean(elapsed_time_results)
    std_elapsed_time = stdev(elapsed_time_results)

    print("Average accuracy:", average_accuracy)
    print("Standard deviation accuracy:", std_accuracy)
    print("Average elapsed time:", average_elapsed_time)
    print("Standard deviation time:", std_elapsed_time)

    return average_accuracy, std_accuracy, average_elapsed_time, std_elapsed_time


if __name__ == "__main__":
    avg_acc, std_acc, avg_time, std_time = multiple_runs(
        5,
        0.5,
        naive_bayes_test,
        "cs4346-image-data/digitdata/traininglabels",
        "cs4346-image-data/digitdata/trainingimages",
        "cs4346-image-data/digitdata/testlabels",
        "cs4346-image-data/digitdata/testimages",
        "digit",
        "raw_pixel"
    )
