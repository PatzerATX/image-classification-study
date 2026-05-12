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

    print(feature_type, image_type, train_percentage)

    return feature_type, train_percentage, average_accuracy, std_accuracy, average_elapsed_time, std_elapsed_time

if __name__ == "__main__":
    all_results = []

    print("\nNAIVE_BAYES_FACE_TESTS")
    for feature in ["raw_pixel", "sliding_window"]:
        for percentage in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            feature_type, train_percentage, avg_acc, std_acc, avg_time, std_time = multiple_runs(
                5,
                percentage,
                naive_bayes_test,
                "cs4346-image-data/facedata/facedatatrainlabels",
                "cs4346-image-data/facedata/facedatatrain",
                "cs4346-image-data/facedata/facedatatestlabels",
                "cs4346-image-data/facedata/facedatatest",
                "face",
                feature
            )
            all_results.append(["naive_bayes", "face", feature_type, train_percentage, avg_acc, std_acc, avg_time, std_time])

    print("\nNAIVE_BAYES_DIGIT_TESTS")
    for feature in ["raw_pixel", "sliding_window"]:
        for percentage in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            feature_type, train_percentage, avg_acc, std_acc, avg_time, std_time = multiple_runs(
                5,
                percentage,
                naive_bayes_test,
                "cs4346-image-data/digitdata/traininglabels",
                "cs4346-image-data/digitdata/trainingimages",
                "cs4346-image-data/digitdata/testlabels",
                "cs4346-image-data/digitdata/testimages",
                "digit",
                feature
            )
            all_results.append(["naive_bayes", "digit", feature_type, train_percentage, avg_acc, std_acc, avg_time, std_time])

    print("\nPERCEPTRON_FACE_TESTS")
    for feature in ["raw_pixel", "sliding_window"]:
        for percentage in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            feature_type, train_percentage, avg_acc, std_acc, avg_time, std_time = multiple_runs(
                5,
                percentage,
                perceptron_test,
                "cs4346-image-data/facedata/facedatatrainlabels",
                "cs4346-image-data/facedata/facedatatrain",
                "cs4346-image-data/facedata/facedatatestlabels",
                "cs4346-image-data/facedata/facedatatest",
                "face",
                feature
            )
            all_results.append(["perceptron", "face", feature_type, train_percentage, avg_acc, std_acc, avg_time, std_time])

    print("\nPERCEPTRON_DIGIT_TESTS")
    for feature in ["raw_pixel", "sliding_window"]:
        for percentage in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            feature_type, train_percentage, avg_acc, std_acc, avg_time, std_time = multiple_runs(
                5,
                percentage,
                perceptron_test,
                "cs4346-image-data/digitdata/traininglabels",
                "cs4346-image-data/digitdata/trainingimages",
                "cs4346-image-data/digitdata/testlabels",
                "cs4346-image-data/digitdata/testimages",
                "digit",
                feature
            )
            all_results.append(["perceptron", "digit", feature_type, train_percentage, avg_acc, std_acc, avg_time, std_time])

    import csv

    with open("all_results.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["algorithm", "dataset", "feature_type", "train_percentage",
                         "avg_accuracy", "std_accuracy", "avg_time", "std_time"
                         ])
        writer.writerows(all_results)

        print("Wrote results to all_results.csv")
