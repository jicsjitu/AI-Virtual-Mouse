import cv2
import mediapipe as mp
import pyautogui
import math
from enum import IntEnum
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from google.protobuf.json_format import MessageToDict
import screen_brightness_control as sbcontrol
import tkinter as tk
from PIL import ImageTk, Image

# Set failsafe for pyautogui
pyautogui.FAILSAFE = False
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Gesture Encodings
class Gest(IntEnum):
    FIST = 0
    PINKY = 1
    RING = 2
    MID = 4
    LAST3 = 7
    INDEX = 8
    FIRST2 = 12
    LAST4 = 15
    THUMB = 16
    PALM = 31
    V_GEST = 33
    TWO_FINGER_CLOSED = 34
    PINCH_MAJOR = 35
    PINCH_MINOR = 36

# Multi-handedness Labels
class HLabel(IntEnum):
    MINOR = 0
    MAJOR = 1

class HandRecog:
    def __init__(self, hand_label):
        self.finger = 0
        self.ori_gesture = Gest.PALM
        self.prev_gesture = Gest.PALM
        self.frame_count = 0
        self.hand_result = None
        self.hand_label = hand_label
    
    def update_hand_result(self, hand_result):
        self.hand_result = hand_result
    
    def get_signed_dist(self, point):
        sign = -1
        if self.hand_result.landmark[point[0]].y < self.hand_result.landmark[point[1]].y:
            sign = 1
        dist = (self.hand_result.landmark[point[0]].x - self.hand_result.landmark[point[1]].x)**2
        dist += (self.hand_result.landmark[point[0]].y - self.hand_result.landmark[point[1]].y)**2
        return math.sqrt(dist) * sign

    def get_dist(self, point):
        dist = (self.hand_result.landmark[point[0]].x - self.hand_result.landmark[point[1]].x)**2
        dist += (self.hand_result.landmark[point[0]].y - self.hand_result.landmark[point[1]].y)**2
        return math.sqrt(dist)

    def get_dz(self, point):
        return abs(self.hand_result.landmark[point[0]].z - self.hand_result.landmark[point[1]].z)

    def set_finger_state(self):
        if self.hand_result is None:
            return
        points = [[8,5,0],[12,9,0],[16,13,0],[20,17,0]]
        self.finger = 0
        self.finger = self.finger | 0 # thumb
        for idx, point in enumerate(points):
            dist = self.get_signed_dist(point[:2])
            dist2 = self.get_signed_dist(point[1:])
            ratio = round(dist / (dist2 or 0.01), 1)
            self.finger = (self.finger << 1) | (1 if ratio > 0.5 else 0)

    def get_gesture(self):
        if self.hand_result is None:
            return Gest.PALM
        current_gesture = Gest.PALM
        if self.finger in [Gest.LAST3, Gest.LAST4] and self.get_dist([8,4]) < 0.05:
            current_gesture = Gest.PINCH_MINOR if self.hand_label == HLabel.MINOR else Gest.PINCH_MAJOR
        elif self.finger == Gest.FIRST2:
            dist1, dist2 = self.get_dist([8, 12]), self.get_dist([5, 9])
            ratio = dist1 / dist2
            current_gesture = Gest.V_GEST if ratio > 1.7 else (Gest.TWO_FINGER_CLOSED if self.get_dz([8, 12]) < 0.1 else Gest.MID)
        else:
            current_gesture = self.finger
        if current_gesture == self.prev_gesture:
            self.frame_count += 1
        else:
            self.frame_count = 0
        self.prev_gesture = current_gesture
        if self.frame_count > 4:
            self.ori_gesture = current_gesture
        return self.ori_gesture

class Controller:
    tx_old = ty_old = 0
    trial = flag = grabflag = pinchmajorflag = pinchminorflag = False
    pinchstartxcoord = pinchstartycoord = pinchdirectionflag = None
    prevpinchlv = pinchlv = framecount = 0
    prev_hand = None
    pinch_threshold = 0.3

    def getpinchylv(hand_result):
        return round((Controller.pinchstartycoord - hand_result.landmark[8].y) * 10, 1)

    def getpinchxlv(hand_result):
        return round((hand_result.landmark[8].x - Controller.pinchstartxcoord) * 10, 1)

    def changesystembrightness():
        currentBrightnessLv = sbcontrol.get_brightness() / 100.0
        currentBrightnessLv = min(max(currentBrightnessLv + Controller.pinchlv / 50.0, 0.0), 1.0)
        sbcontrol.fade_brightness(int(100 * currentBrightnessLv), start=sbcontrol.get_brightness())

    def changesystemvolume():
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        currentVolumeLv = min(max(volume.GetMasterVolumeLevelScalar() + Controller.pinchlv / 50.0, 0.0), 1.0)
        volume.SetMasterVolumeLevelScalar(currentVolumeLv, None)

    def scrollVertical():
        pyautogui.scroll(120 if Controller.pinchlv > 0 else -120)

    def scrollHorizontal():
        pyautogui.keyDown('shift')
        pyautogui.keyDown('ctrl')
        pyautogui.scroll(-120 if Controller.pinchlv > 0 else 120)
        pyautogui.keyUp('ctrl')
        pyautogui.keyUp('shift')

    def get_position(hand_result):
        sx, sy = pyautogui.size()
        x, y = int(hand_result.landmark[9].x * sx), int(hand_result.landmark[9].y * sy)
        Controller.prev_hand = [x, y] if Controller.prev_hand is None else Controller.prev_hand
        delta_x, delta_y = x - Controller.prev_hand[0], y - Controller.prev_hand[1]
        dist_sq = delta_x ** 2 + delta_y ** 2
        ratio = 0 if dist_sq <= 25 else (0.07 * (dist_sq ** 0.5) if dist_sq <= 900 else 2.1)
        Controller.prev_hand = [x, y]
        return int(x + delta_x * ratio), int(y + delta_y * ratio)

    def pinch_control_init(hand_result):
        Controller.pinchstartxcoord = hand_result.landmark[8].x
        Controller.pinchstartycoord = hand_result.landmark[8].y
        Controller.pinchlv = Controller.prevpinchlv = Controller.framecount = 0

    def pinch_control(hand_result, controlHorizontal, controlVertical):
        if Controller.framecount == 5:
            Controller.pinchlv = Controller.prevpinchlv
            (controlHorizontal if Controller.pinchdirectionflag else controlVertical)()
            Controller.framecount = 0
        lvx, lvy = Controller.getpinchxlv(hand_result), Controller.getpinchylv(hand_result)
        if abs(lvy) > abs(lvx) and abs(lvy) > Controller.pinch_threshold:
            Controller.pinchdirectionflag, Controller.framecount = False, Controller.framecount + 1 if abs(Controller.prevpinchlv - lvy) < Controller.pinch_threshold else 0
            Controller.prevpinchlv = lvy
        elif abs(lvx) > Controller.pinch_threshold:
            Controller.pinchdirectionflag, Controller.framecount = True, Controller.framecount + 1 if abs(Controller.prevpinchlv - lvx) < Controller.pinch_threshold else 0
            Controller.prevpinchlv = lvx

    def handle_controls(gesture, hand_result):
        x, y = (Controller.get_position(hand_result) if gesture != Gest.PALM else (None, None))
        if gesture == Gest.V_GEST:
            Controller.flag = True
            pyautogui.moveTo(x, y, duration=0.1)
        elif gesture == Gest.FIST and not Controller.grabflag:
            Controller.grabflag = True
            pyautogui.mouseDown(button="left")
            pyautogui.moveTo(x, y, duration=0.1)
        elif gesture == Gest.MID and Controller.flag:
            pyautogui.click()
            Controller.flag = False
        elif gesture == Gest.INDEX and Controller.grabflag:
            Controller.grabflag = False
            pyautogui.mouseUp(button="left")
        elif gesture == Gest.TWO_FINGER_CLOSED and Controller.pinchmajorflag:
            Controller.pinch_control_init(hand_result)
            Controller.pinchmajorflag = False
            Controller.trial = True
        elif gesture == Gest.PINCH_MAJOR and Controller.trial:
            Controller.pinch_control(hand_result, Controller.scrollHorizontal, Controller.scrollVertical)
        elif gesture == Gest.PINCH_MINOR and not Controller.pinchminorflag:
            Controller.pinch_control_init(hand_result)
            Controller.pinchminorflag = True
        elif gesture == Gest.PINCH_MINOR and Controller.pinchminorflag:
            Controller.pinch_control(hand_result, Controller.changesystemvolume, Controller.changesystembrightness)

class GestureController:
    gc_mode = True
    cap = None

    def __init__(self):
        self.main_hand = None
        self.hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
        self.running = False  # Track whether the GestureController is running

    def classify_hands(self, results):
        if results.multi_handedness:
            label = MessageToDict(results.multi_handedness[0])["classification"][0]["label"]
            self.main_hand = HandRecog(HLabel.MAJOR if label == "Right" else HLabel.MINOR)

    def start(self):
        self.cap = cv2.VideoCapture(0)
        self.running = True  # Set running to True to start tracking

        while self.cap.isOpened() and self.running:  # Check self.running in the loop
            success, image = self.cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip and process the image
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            results = self.hands.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Check if hands are detected
            if results.multi_hand_landmarks:
                if not self.main_hand:
                    self.classify_hands(results)

                for hand_landmarks in results.multi_hand_landmarks:
                    if self.main_hand:
                        self.main_hand.update_hand_result(hand_landmarks)
                        self.main_hand.set_finger_state()
                        gesture = self.main_hand.get_gesture()
                        Controller.handle_controls(gesture, hand_landmarks)
                    mp.solutions.drawing_utils.draw_landmarks(image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

            cv2.imshow("Virtual Mouse", image)
            if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def stop(self):
        self.running = False  # Set running to False to stop the loop

