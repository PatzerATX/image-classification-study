from raw_pixel_feature_extraction import raw_pixel_feature_extraction

def sliding_window_feature_extraction(label_path, image_path):
    WINDOW_SIZE = 3
    SLIDE = 2

    raw_pixel_data, image_height, image_width = raw_pixel_feature_extraction(label_path, image_path)

    sliding_window_feature_vector = []

    for label, raw_pixel_vector in raw_pixel_data:
        sliding_window_data = []

        for start_row in range(0, image_height - WINDOW_SIZE + 1, SLIDE):
            for start_col in range(0, image_width - WINDOW_SIZE + 1, SLIDE):
                count = 0
    
                for row in range(start_row, start_row + WINDOW_SIZE):
                    for col in range(start_col, start_col + WINDOW_SIZE):
                        index = row * image_width + col
                        count += raw_pixel_vector[index]
                if count > 3:
                    count = 1
                else:
                    count = 0
                sliding_window_data.append(count)

        sliding_window_feature_vector.append((label, sliding_window_data))

    return sliding_window_feature_vector



