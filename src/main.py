from omegaconf import OmegaConf, DictConfig
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import json
import os
from transport import Track_transport, Zone
from utils import get_crop_x, is_inside_zone
import hydra


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(cfg: DictConfig):
    '''
    Основной алгоритм программы

    Args:
        cfg (DictConfig): файл с параметрами системы
    ''' 
    print(OmegaConf.to_yaml(cfg))
    COLUMNS = cfg.params.COLUMNS
    SKIPPED_FRAMES = cfg.params.SKIPPED_FRAMES
    VIDEO_DIR = cfg.params.VIDEOS_DIR
    JSON_DIR = cfg.params.JSONS_DIR
    CLASSES = cfg.params.CLASSES
    if not os.path.exists('result'):
        os.mkdir('result')
    result_df = pd.DataFrame(columns=COLUMNS)
    videos = os.listdir(VIDEO_DIR)
    print('Перечень доступных видео для обработки: ', videos)
    model = YOLO('yolov8s.pt') #Импорт модели
    #for video_frame_idx in range(len(videos) - 1):
    for i, video_name in enumerate(videos):
        #video_frame = videos[video_frame_idx]
        json_name = video_name.split('.mp4')[0]
        if json_name == '.DS_Store':
            continue
        with open(f'{JSON_DIR}/{json_name}.json', 'r', encoding='utf-8') as f:
            area_markup = json.load(f)
        zone = Zone(area_markup)
        video_path = f'{VIDEO_DIR}/{video_name}'
        #----------------------------------------
        #Настройка параметров обработки видео
        video_capture = cv2.VideoCapture(video_path)
        video_fps = video_capture.get(cv2.CAP_PROP_FPS)
        print(f'=== {video_name} - {video_fps} FPS ===')
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
        size = (width, height)
        #out = cv2.VideoWriter(f'result/result_{video_name}.mp4', fourcc, video_fps, size)
        #----------------------------------------
        transport_dict: dict = {}
        #appeared_ids = []
        frame_idx = 0
        frame_idx_in_cond = 0
        while video_capture.isOpened():
            success, frame = video_capture.read()

            if success:
                if frame_idx % SKIPPED_FRAMES == 0: # пропускаем некоторые кадры для ускорения обработки
                    frame = cv2.resize(frame, (640, 360)) # (1280, 720)
                    track_results = model.track(frame,
                                                persist=True,
                                                classes=CLASSES,
                                                verbose=False,
                                                conf=0.5,
                                                imgsz=(384, 640)) 
                    annotated_frame = track_results[0].plot()
                    try:
                        frame_ids = track_results[0].boxes.id.numpy()
                        frame_cls = track_results[0].boxes.cls.numpy()
                        bb_center = track_results[0].boxes.xywh.numpy()
                        bb_corners = track_results[0].boxes.xyxy.numpy()
                        #print(type(frame_ids), type(frame_cls))

                        #Реализация треккинга
                        for i, idx in enumerate(frame_ids):

                            #idx = str(int(frame_ids[i]))
                            idx = str(int(idx))

                            #если такого транспорта еще не было - добавить в словарь его номер
                            if idx not in transport_dict.keys():
                                transport_dict[idx] = Track_transport(str(idx))
                                
                            #отображение центра объекта синим кружком
                            cv2.circle(annotated_frame, (int(bb_center[i][0]), int(bb_center[i][1])), 6, (255, 0, 0), -1)

                            #если объект находится в зоне
                            if zone.is_inside_zone(bb_center[i][0], bb_center[i][1] + (bb_center[i][3] / 2), annotated_frame):
                                transport_dict[idx].frames.append(frame_idx_in_cond)
                                transport_dict[idx].type.append(int(frame_cls[i]))
                                transport_dict[idx].inside_area = True
                                #Отрисовываем только те bb, которые попадают в зону
                                '''[x1, y1, x2, y2] = bb_corners[i]
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                                
                                cv2.putText(frame, f'{int(rame_cls[i])}')
                                frame = cv2.resize(frame, (1280, 640))'''
                    

                    except Exception as e:
                        print(e)

                    annotated_frame = cv2.resize(annotated_frame, size)
                    cv2.imshow('Transport tracking', annotated_frame)
                    #out.write(annotated_frame)
                    key = cv2.waitKey(1)
                    frame_idx_in_cond += 1

                    if key == 30:
                        break
                frame_idx += 1
            else:
                break

        class_average_speed: dict = {2: [], 5: [], 7: []}
        count_dict = dict.fromkeys(CLASSES, 0)

        for k in transport_dict.keys():
            ts = transport_dict[k]

            if len(ts.frames) < video_fps//4:
                continue

            d = dict.fromkeys(ts.type)
            for k in d.keys():
                d[k] = ts.type.count(k)
            type = max(d, key=d.get)

            if ts.inside_area:
                count_dict[type] += 1

            if ts.inside_area:
                my_fps  = (video_fps / SKIPPED_FRAMES) # количество кадров, где был замечен объект
                seconds = (ts.frames[-1] - ts.frames[0] + 1) / my_fps #кол-во секунд = кол-во кадров / fps
                m = 20 #кол-во метров между конусами 
                #перевод скорости из м/с в км/ч: v_km_h = (v_m_s * 3600) / 1000 = v_m_s * 3.6
                average_speed = 3.6 * ( m / seconds)
                #print(f'Average_speed = {average_speed}')
                class_average_speed[type].append(average_speed)

        for k in class_average_speed.keys():
            class_average_speed[k] = sum(class_average_speed[k]) / len(class_average_speed[k]) if sum(class_average_speed[k]) != 0 else 0

        print(f'Средняя скорость по классу: {class_average_speed}')
        print(f'Финальное количество объектов в видео {video_name}: {count_dict}')
        file_name = video_name.split('.')[0]

        try:
            car_avg_spd = class_average_speed[2]
        except KeyError: 
            car_avg_spd = 0
        
        try:
            bus_avg_spd = class_average_speed[5]
        except KeyError: 
            bus_avg_spd = 0
        
        try:
            van_avg_spd = class_average_speed[7]
        except KeyError: 
            van_avg_spd = 0

        result = [file_name, count_dict[2], car_avg_spd, count_dict[5], bus_avg_spd, count_dict[7], van_avg_spd]

        result_df.loc[i] = result

        video_capture.release()
        cv2.destroyAllWindows()

    print(result_df)
    result_df.to_csv('result/submission.csv', index=False)


if __name__ == '__main__':
    main()
