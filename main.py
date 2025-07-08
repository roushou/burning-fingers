import cv2
from cv2.typing import MatLike
import mediapipe as mp
import random

# Constants
MAX_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.8
FINGERTIP_INDICES = [8, 12]  # Index, Middle
FLAME_LIFE = 50
FLAME_SIZE = 10
FLAME_VY_RANGE = (-20, 0)


class Flame:
    def __init__(self, x: float, y: float):
        self.x: float = x
        self.y: float = y
        self.vy: float = random.uniform(FLAME_VY_RANGE[0], FLAME_VY_RANGE[1])
        self.life: int = FLAME_LIFE

    def update(self) -> bool:
        self.y += self.vy
        self.life -= 1
        return self.life > 0

    def draw(self, frame: MatLike) -> None:
        alpha = self.life / FLAME_LIFE
        color = (0, int(alpha * 165), int(alpha * 255))  # Red to yellow
        _ = cv2.circle(frame, (int(self.x), int(self.y)), FLAME_SIZE, color, -1)


def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=MAX_HANDS, min_detection_confidence=MIN_DETECTION_CONFIDENCE
    )

    video_capture = cv2.VideoCapture(0)
    flames: list[Flame] = []

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        frame = cv2.flip(frame, 1)
        frame_height, frame_width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for idx in FINGERTIP_INDICES:
                    landmark = hand_landmarks.landmark[idx]
                    x, y = int(landmark.x * frame_width), int(landmark.y * frame_height)
                    flames.append(Flame(x, y))

        flames[:] = [f for f in flames if f.update()]
        for flame in flames:
            flame.draw(frame)

        cv2.imshow("Burning Fingers", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
