import time

from naive_bayes_train import naive_bayes_train
from naive_bayes_prediction import naive_bayes_prediction
from raw_pixel_feature_extraction import raw_pixel_feature_extraction
from sliding_window_feature_extraction import sliding_window_feature_extraction


def naive_bayes_test(training_label_path, training_image_path,
                           validation_label_path, validation_image_path,
                         image_type, feature_type, train_percentage):

    start_time = time.perf_counter()

    label_probabilities, feature_probabilities = (naive_bayes_train(
        training_label_path,
        training_image_path,
        image_type,
        feature_type,
        train_percentage
    ))

    if feature_type == "raw_pixel":
        test_data, _, _ = raw_pixel_feature_extraction(validation_label_path, validation_image_path)
    elif feature_type == "sliding_window":
        test_data = sliding_window_feature_extraction(validation_label_path, validation_image_path)

    correct_count = 0

    for actual_label, image_data in test_data:
        predicted_label = naive_bayes_prediction(label_probabilities, feature_probabilities, image_data)

        if actual_label == predicted_label:
            correct_count += 1

    prediction_accuracy = correct_count / len(test_data)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    return prediction_accuracy, elapsed_time

if __name__ == "__main__":
    for feature_type in ["raw_pixel", "sliding_window"]:
        for percentage in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            prediction_accuracy = naive_bayes_test(
                "cs4346-image-data/digitdata/traininglabels",
                "cs4346-image-data/digitdata/trainingimages",
                "cs4346-image-data/digitdata/testlabels",
                "cs4346-image-data/digitdata/testimages",
                "digit",
                feature_type,
                percentage
            )

            print ("Naive Bayes", feature_type, "digit success at", percentage,":" , prediction_accuracy)

        for percentage in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            prediction_accuracy = naive_bayes_test(
            "cs4346-image-data/facedata/facedatatrainlabels",
            "cs4346-image-data/facedata/facedatatrain",
            "cs4346-image-data/facedata/facedatatestlabels",
            "cs4346-image-data/facedata/facedatatest",
            "face",
            feature_type,
            percentage
        )

            print("Naive Bayes", feature_type, "face success at", percentage, ":", prediction_accuracy)
