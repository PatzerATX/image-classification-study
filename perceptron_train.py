from raw_pixel_feature_extraction import raw_pixel_feature_extraction
from sliding_window_feature_extraction import sliding_window_feature_extraction

import random

def perceptron_train(label_path, image_path, image_type, feature_type, train_percentage):

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

    num_feature_values = len(picture_data[0][1]) #length of features vector

    # each label value has a vector of weights
    label_weights = [[0] * (num_feature_values + 1) for label in range(num_label_values)]

    weights_updated = True
    training_run = 0

    while (weights_updated and training_run < 20):
        weights_updated = False
        for actual_label, image_data in picture_data: #loop through each image
            for possible_label in range(num_label_values): #loop through each possible label value for each image
                label_score = label_weights[possible_label][0]
                for value in range(num_feature_values): #loop through each feature value in the list
                    label_score += label_weights[possible_label][value + 1] * image_data[value]

                if actual_label == possible_label:
                    if label_score < 0:
                        for value in range(num_feature_values):
                            label_weights[possible_label][value + 1] += image_data[value]
                        label_weights[possible_label][0] += 1
                        weights_updated = True

                else:
                    if label_score >= 0:
                        for value in range(num_feature_values):
                            label_weights[possible_label][value + 1] -= image_data[value]
                        label_weights[possible_label][0] -= 1
                        weights_updated = True

        training_run += 1

    print("training runs:", training_run)
    return label_weights

import time

if __name__ == "__main__":
    start = time.perf_counter()

    label_weights = perceptron_train(
        "cs4346-image-data/digitdata/traininglabels",
        "cs4346-image-data/digitdata/trainingimages",
        "digit",
        "raw_pixel",
        0.5
    )

    elapsed = time.perf_counter() - start
    print("training time:", elapsed)


