from utils import DrowsinessDetector

def main():
    driver_info = ''
    shape_predictor = ''
    url = ''
    webcam = 0
    
    drowsy = DrowsinessDetector(driver_info, shape_predictor, url, webcam)
    drowsy.execute()

if __name__ == '__main__':
    main()