from pathlib import Path

def extract_and_label_data(label_path, image_path):
    labeled_image_vector = []
    label_vector = []
    label_lines = Path(label_path).read_text().splitlines()

    for line in label_lines:
        label_vector.append(int(line.strip()))

    image_lines = Path(image_path).read_text().splitlines()
    image_height = len(image_lines) // len(label_lines)
    image_width = len(image_lines[0])

    label_count = 0
    for image in range(0, len(image_lines), image_height):
        one_image_lines = image_lines[image : image + image_height]

        one_image_vector = []

        for line in one_image_lines:
            for char in line:
                one_image_vector.append(char)

        labeled_image_vector.append((label_vector[label_count], one_image_vector))
        label_count += 1

    return labeled_image_vector, image_height, image_width



