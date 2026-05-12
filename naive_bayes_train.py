from raw_pixel_feature_extraction import raw_pixel_feature_extraction
from sliding_window_feature_extraction import sliding_window_feature_extraction

import random

def naive_bayes_train(label_path, image_path, image_type, feature_type, train_percentage):

    if image_type == "face":
        num_label_values = 2
    elif image_type == "digit":
        num_label_values = 10
    else:
        raise ValueError("Unknown image type")

    if feature_type == "raw_pixel":
        picture_data, _, _ = raw_pixel_feature_extraction(label_path, image_path)
    elif feature_type == "sliding_window":
        picture_data = sliding_window_feature_extraction(label_path, image_path)
    else:
        raise ValueError("Unknown feature type")

    num_training_images = int(len(picture_data) * train_percentage)
    picture_data = random.sample(picture_data, num_training_images)

    num_feature_values = len(picture_data[0][1])

    label_values_count = [0] * num_label_values
    feature_values_count = [[0] * num_feature_values for label_value in range(num_label_values)]

    for label, feature_values in picture_data:
        label_values_count[label] += 1
        for value in range(num_feature_values):
            feature_values_count[label][value] += feature_values[value]

    #Calculate P(y) for each label_value
    label_probabilities = [0] * num_label_values

    for label in range(num_label_values):
        label_probabilities[label] = label_values_count[label] / len(picture_data)

    #Calculate P(x_i = 1 | y)
    feature_probabilities = [[0] * num_feature_values for label_value in range(num_label_values)]

    for label in range(num_label_values):
        for feature in range(num_feature_values):
            feature_probabilities[label][feature] = ((feature_values_count[label][feature] + 0.1) /
                                                     (label_values_count[label] + 0.2))

    return label_probabilities, feature_probabilities

import time

if __name__ == "__main__":
    start = time.perf_counter()

    label_weights = naive_bayes_train(
        "cs4346-image-data/digitdata/traininglabels",
        "cs4346-image-data/digitdata/trainingimages",
        "digit",
        "raw_pixel",
        1.0
    )

    elapsed = time.perf_counter() - start

    print("training time:", elapsed)
