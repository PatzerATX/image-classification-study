from data_extraction import extract_and_label_data

def raw_pixel_feature_extraction(label_path, image_path):
    raw_pixel_feature_vector = []
    data_vector, image_height, image_width = extract_and_label_data(label_path, image_path)

    for label, image in data_vector:
        raw_pixel_vector = []
        for char in image:
            if char == " ":
                raw_pixel_vector.append(0)
            else:
                raw_pixel_vector.append(1)
        raw_pixel_feature_vector.append((label, raw_pixel_vector))

    return raw_pixel_feature_vector, image_height, image_width
