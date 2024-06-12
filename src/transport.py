import cv2
import numpy as np


class Track_transport:
    '''
    Класс для треккинга объектов,
    который:
            - находится ли транспорт в зоне замера средней скорости,
            - к какому классу транспорта относится объект,
            - какой номер у объекта (для треккинга),
            - на каких кадрах он был зафиксирован
    '''
    def __init__(self, idx: str) -> None:
        self.frames: list = []
        self.inside_area: bool = False
        self.idx: str = idx
        self.type: list = []

class Zone:
    '''
    Класс, реализующий методы для обработки разметки зоны
    '''
    
    def __init__(self, area_markup: dict):
        self.area_markup = area_markup

    def is_inside_zone(self, point_x, point_y, frame) -> bool:
        '''
        Функция проверки находится ли объект внутри зоны замера средней скорости

        Args:
            point_x (_type_): x координата центра объекта
            point_y (_type_): y координата центра объекта
            frame (_type_): кадр

        Returns:
            bool: Флаг нахождения в зоне (True, если объект находится ВНУТРИ зоны, False - ВНЕ зоны)
        '''
        areas = []
        areas1 = self.area_markup['areas'][0]

        p1 = int(areas1[0][0] * frame.shape[1]), int(areas1[0][1] * frame.shape[0])
        p2 = int(areas1[1][0] * frame.shape[1]), int(areas1[1][1] * frame.shape[0])
        p3 = int(areas1[2][0] * frame.shape[1]), int(areas1[2][1] * frame.shape[0])
        p4 = int(areas1[3][0] * frame.shape[1]), int(areas1[3][1] * frame.shape[0])

        areas.append(np.array([p1, p2, p3, p4]))

        try:
            areas2 = self.area_markup['areas'][1]
            p1 = int(areas2[0][0] * frame.shape[1]), int(areas2[0][1] * frame.shape[0])
            p2 = int(areas2[1][0] * frame.shape[1]), int(areas2[1][1] * frame.shape[0])
            p3 = int(areas2[2][0] * frame.shape[1]), int(areas2[2][1] * frame.shape[0])
            p4 = int(areas2[3][0] * frame.shape[1]), int(areas2[3][1] * frame.shape[0])
            areas.append(np.array([p1, p2, p3, p4]))
        except:
            pass
        for area in areas:
            #отображение зон на кадре
            cv2.polylines(frame, pts=[area], isClosed=True, color=(0, 0, 255))
            #поиск наименьшего расстояния между точкой и областью
            #положительное d - если точка внутри области
            #равно 0 - если точка лежит на контуре области
            if cv2.pointPolygonTest(area, (point_x, point_y), measureDist=False) >= 0:
                return True
        return False