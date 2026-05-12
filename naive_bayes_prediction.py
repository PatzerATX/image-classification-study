import math

def naive_bayes_prediction(label_probabilities, feature_probabilities, image_prediction_data):
    high_value = float("-inf")
    high_label = 0

    for label in range(len(label_probabilities)):
        current_score = math.log(label_probabilities[label]) #log(0) -> log(1) = -inf -> 0

        for feature in range(len(feature_probabilities[label])):
            if image_prediction_data[feature] == 1:
                current_score += math.log(feature_probabilities[label][feature]) #logs of results are summed
            else:
                current_score += math.log(1 - feature_probabilities[label][feature]) #accounts for 0s since binary data

        if current_score > high_value:
            high_value = current_score
            high_label = label

    return high_label
