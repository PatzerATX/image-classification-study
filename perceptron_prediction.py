def perceptron_prediction(label_weights, image_prediction_data):
    high_label = 0
    high_score = float("-inf")

    for label in range(len(label_weights)):
        label_score = label_weights[label][0]

        for weight in range(len(image_prediction_data)):
            label_score += label_weights[label][weight + 1] * image_prediction_data[weight]

        if label_score > high_score:
            high_score = label_score
            high_label = label

    return high_label
