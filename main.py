import cv2
import json
import argparse
import os
from ultralytics import YOLO
import pandas as pd

# Константы
# Модель YOLO
MODEL_NAME = "yolov8n.pt"

# Пути
ROI_PATH = "table_roi.json"
OUTPUT_VIDEO_PATH = "output/output.mp4"
OUTPUT_EVENTS_PATH = "output/events.parquet"
OUTPUT_EVENTS_TIMELINE_PATH = "output/events_timeline.parquet"
OUTPUT_REPORT_PATH = "output/report.txt"

# визуал
MAX_DISPLAY_WIDTH = 1280
MAX_DISPLAY_HEIGHT = 720
WINDOW_NAME = "Table Detection"
TEXT_SCALE = 0.9
TEXT_THICKNESS = 2
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
COLOR_EMPTY = (0, 255, 0)
COLOR_OCCUPIED = (0, 0, 255)

# Для детекции
INIT_DURATION_SEC = 2.0       # 2 сек для старта анализа
BUFFER_DURATION_SEC = 10.0     # 2 сек что сгладить сработки
MIN_STABLE_DURATION_SEC = 20.0 # небольшой гап чтобы убедится что кто-то ушел либо пришел
MIN_ABSENCE_DURATION = 20.0    # минимальное время на то что кто-то ушел
OCCUPANCY_THRESHOLD = 0.6     # 60% кадров для сглаживания
INITIAL_THRESHOLD = 0.5       # 50% кадров для начального состояния


def parse_args():
    """
    Парсим параметры
    """
    parser = argparse.ArgumentParser(description="Детекция уборки столиков по видео")
    parser.add_argument("--video", type=str, default="src_videos/video_2.mp4", help="Путь к видеофайлу")
    parser.add_argument("--duration", type=int, default=-1, help="Макс. длительность обработки в секундах (-1 = всё видео)")
    parser.add_argument("--silent_mode", action="store_true", help="Режим без визуализации")
    parser.add_argument("--select-roi", action="store_true", help="Запустить интерактивный выбор ROI")
    return parser.parse_args()

def select_table_zone(video_path):
    """
    Выбор зоны стола.
    Показываем первый кадр чтобы выбрать ROI
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError("Не удалось прочитать видео")

    roi = cv2.selectROI("Select table zone", frame, False)
    cv2.destroyWindow("Select table zone")

    roi_dict = {"x": int(roi[0]), "y": int(roi[1]), "w": int(roi[2]), "h": int(roi[3])}
    with open(ROI_PATH, "w") as f:
        json.dump(roi_dict, f)

    print(f"Выбранный ROI: {roi_dict}")


def is_person_in_roi(results, roi):
    """
    Проверяет, есть ли человек в ROI.
    """
    x1_r, y1_r, w_r, h_r = roi
    x2_r, y2_r = x1_r + w_r, y1_r + h_r

    for result in results:
        boxes = result.boxes
        for box in boxes:
            if box.cls == 0:  # класс из модели 0 = человек
                xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                cx = int((xyxy[0] + xyxy[2]) / 2)
                cy = int((xyxy[1] + xyxy[3]) / 2)

                if x1_r < cx < x2_r and y1_r < cy < y2_r:
                    return True
    return False

def draw_visuals(frame, roi, person_detected):
    """
    Рисует ROI и статус на кадре
    """
    x, y, w, h = roi
    color = COLOR_EMPTY if not person_detected else COLOR_OCCUPIED
    thickness = 3

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    status = "Occupied" if person_detected else "Empty"
    cv2.putText(frame, f"Table 15: {status}", (x, y - 10),
                FONT_FACE, TEXT_SCALE, color, TEXT_THICKNESS)

def process_video(video_path, duration=None, silent_mode=False):
    """
    Основная функция обработки видео
    возвращает датафрейм с собитиями
    """
    if not os.path.exists(ROI_PATH):
        print(f"ROI не найден. Сначала запустите с --select-roi")
        return None

    with open(ROI_PATH, "r") as f:
        roi_data = json.load(f)
        roi = (roi_data["x"], roi_data["y"], roi_data["w"], roi_data["h"])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Не удалось открыть видео")
    
    # инициализируем модель
    model = YOLO(MODEL_NAME)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Подготовка VideoWriter
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # если указал длину видео
    max_frames = None
    if duration is not None:
        max_frames = int(duration * fps)

    events = []
    frame_count = 0

    # Так как из-за попыток сгладить мелькание модели при уходе чела с видео 2,
    # на старте не сразу правильно определяется состояние
    # то по тихому анализируем первые 2 секунды видео
    init_frames = int(INIT_DURATION_SEC * fps)
    init_history = []

    for _ in range(init_frames):
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, verbose=False)
        raw_in_roi = is_person_in_roi(results, roi)
        init_history.append(raw_in_roi)

    # Определяем начальное состояние: если человек был в >50% кадров — считаем "занято"
    initial_occupied = sum(init_history) >= len(init_history) * INITIAL_THRESHOLD
    stable_occupied = initial_occupied
    last_state_change_frame = 0

    # матаем на начало
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Буфер для сглаживания состояния
    history = []
    buffer_size = int(BUFFER_DURATION_SEC * fps)

    # Минимальное время для смены состояния
    min_stable_duration_frames = int(MIN_STABLE_DURATION_SEC * fps)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Видео закончилось")
            break
        if max_frames is not None and frame_count >= max_frames:
            break

        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

        # Детекция людей
        results = model(frame, verbose=False)
        raw_in_roi = is_person_in_roi(results, roi)

        # Добавляем в историю
        history.append(raw_in_roi)
        if len(history) > buffer_size:
            history.pop(0)

        # Сглаживаем: если человек был виден хотя бы в 60% кадров
        occupied_smooth = sum(history) >= len(history) * OCCUPANCY_THRESHOLD

        # Проверяем, можно ли менять состояние
        state_changed = occupied_smooth != stable_occupied
        frames_since_change = frame_count - last_state_change_frame

        if state_changed and frames_since_change >= min_stable_duration_frames:
            stable_occupied = occupied_smooth
            last_state_change_frame = frame_count

        # Используем стабильное состояние
        person_in_roi = stable_occupied

        # Сохраняем событие
        events.append({
            "frame": frame_count,
            "time": round(timestamp, 3),
            "occupied": person_in_roi,
            "raw_detected": raw_in_roi  # для отладки
        })

        # Рисуем
        draw_visuals(frame, roi, person_in_roi)

        # Записываем кадр
        out.write(frame)

        # Показываем окно если не указан тихий режим
        if not silent_mode:
            # только раз создаем окно
            if frame_count == 0:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WINDOW_NAME, MAX_DISPLAY_WIDTH, MAX_DISPLAY_HEIGHT)

            # Немножко некрасивостей чтобы смасштабировать видос
            h, w = frame.shape[:2]
            scale = min(MAX_DISPLAY_WIDTH / w, MAX_DISPLAY_HEIGHT / h)
            if scale < 1:
                new_size = (int(w * scale), int(h * scale))
                display_frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
            else:
                display_frame = frame

            cv2.imshow(WINDOW_NAME, display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_count += 1

    cap.release()
    out.release()
    if not silent_mode:
        cv2.destroyAllWindows()

    df = pd.DataFrame(events)
    df.to_parquet(OUTPUT_EVENTS_PATH, index=False, engine="pyarrow", compression=None)
    print(f"Видео сохранено: {OUTPUT_VIDEO_PATH}")

    return df

def analyze_events(df):
    """
    Анализирует события: находит освобождения и подходы,
    считает задержку между ними.
    """
    events_log = []
    last_cleared_time = None

    for i in range(1, len(df)):
        prev_state = df.iloc[i-1]["occupied"]
        curr_state = df.iloc[i]["occupied"]
        timestamp = df.iloc[i]["time"]

        # Немножко дичи для оперделения реальных подходов
        # Событие: стол освободился (занят → пуст)
        if prev_state and not curr_state:
            events_log.append({
                "event": "cleared",
                "time": timestamp,
                "delay": None
            })
            last_cleared_time = timestamp

        # Событие: к столу подошли (пуст → занят)
        elif not prev_state and curr_state:
            # Проверяем, был ли длительный перерыв (реальный подход)
            is_real_approach = (
                last_cleared_time is not None and
                (timestamp - last_cleared_time) >= MIN_ABSENCE_DURATION
            )

            event_entry = {
                "event": "approached" if is_real_approach else "returned",
                "time": timestamp,
            }

            if is_real_approach:
                delay = timestamp - last_cleared_time
                event_entry["delay"] = delay
                events_log.append(event_entry)
                last_cleared_time = None  # сбрасываем после подхода
            else:
                # Человек вернулся быстро — не "подход", а "возврат"
                event_entry["delay"] = timestamp - last_cleared_time if last_cleared_time else None
                events_log.append(event_entry)

    if not events_log:
        print("Нет событий для анализа")
        return pd.DataFrame(columns=["event", "time", "delay"]), None

    events_df = pd.DataFrame(events_log)

    # Считаем среднее только по реальным подходам
    avg_delay = None
    approached_events = events_df[events_df["event"] == "approached"]
    if not approached_events.empty:
        avg_delay = approached_events["delay"].mean()

    return events_df, avg_delay

def main():
    args = parse_args()

    # Выбор ROI
    if args.select_roi:
        select_table_zone(args.video)
        return

    # Обработка видео
    duration = None if args.duration == -1 else args.duration
    df = process_video(args.video, duration=duration, silent_mode=args.silent_mode)

    if df is None or len(df) == 0:
        return

    print(f"Среднее состояние: {df['occupied'].mean():.2%} времени занят")

    # Типо аналитика
    events_df, avg_delay = analyze_events(df)

    # Сохраняем лог событий, чтобы было
    events_df.to_parquet(OUTPUT_EVENTS_TIMELINE_PATH, index=False, engine="pyarrow", compression=None)

    # Вывод результата
    approached_count = len(events_df[events_df["event"] == "approached"])
    cleared_count = len(events_df[events_df["event"] == "cleared"])

    print(f"Освобождений стола: {cleared_count}")
    print(f"Подходов к столу: {approached_count}")
    if avg_delay is not None:
        print(f"Среднее время задержки: {avg_delay:.2f} сек")
    else:
        print("Не найдено пар 'освобождение → подход'")

    # Сохраняем отчёт
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(f"Отчёт по видео: {args.video}\n")
        f.write(f"Обработано: {len(df)} кадров\n")
        f.write(f"Освобождений: {cleared_count}\n")
        f.write(f"Подходов: {approached_count}\n")
        f.write(f"Среднее время задержки: {avg_delay:.2f} сек\n" if avg_delay else "Среднее время: недостаточно данных\n")
    print(f"Отчёт: {OUTPUT_REPORT_PATH}")

if __name__ == "__main__":
    main()
