import os
import numpy as np
import cv2

def create_dataset(folder_path, output_file):
    dataset = {
        'imgs': [],
        'emotionl1': [],
        'emotionl2': [],
        'label': []
    }

    cnt = 0;

    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
            img_path = os.path.join(folder_path, filename)
            img_path = os.path.abspath(img_path)
            print(f"이미지 로드 시도: {img_path}")
            cnt += 1
            if os.path.isfile(img_path):
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        img_resized = cv2.resize(img, (224, 224))
                        dataset['imgs'].append(img_resized)
                        dataset['emotionl1'].append(6)
                        dataset['emotionl2'].append(6)
                        dataset['label'].append(6)
                    else:
                        print(f"이미지 로드 오류 {img_path}: 이미지가 None입니다")
                except cv2.error as e:
                    print(f"OpenCV 에러로 이미지 로드 실패 {img_path}: {e}")
                except Exception as e:
                    print(f"이미지 로드 중 예외 발생 {img_path}: {e}")
            else:
                print(f"파일이 존재하지 않습니다: {img_path}")

            if cnt == 2000:
                break

    dataset['imgs'] = np.array(dataset['imgs'])
    dataset['emotionl1'] = np.array(dataset['emotionl1'])
    dataset['emotionl2'] = np.array(dataset['emotionl2'])
    dataset['label'] = np.array(dataset['label'])

    np.save(output_file, dataset)

# 사용 예시
if __name__ == "__main__":
    folder_path = '' #저장 위치
    output_file = 'wound.npy'

    create_dataset(folder_path, output_file)
