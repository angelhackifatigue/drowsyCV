import json
from pathlib import Path

from utils.DrowsinessDetector import DrowsinessDetector


def main():
    driver_path = Path.cwd() / 'utils' / 'driver_info.json'
    json_file = open(driver_path)
    driver_info = json.load(json_file)
    shape_predictor = str(Path.cwd() / 'detectors' / 'shape_predictor_68_face_landmarks.dat')
    url = 'https://drowsiness-detector-d7660.firebaseio.com/dangers.json'
    webcam = 0

    drowsy = DrowsinessDetector(driver_info, shape_predictor, url, webcam)
    drowsy.execute()

if __name__ == '__main__':
    main()
