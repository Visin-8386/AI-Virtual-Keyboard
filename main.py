import cv2
import numpy as np
import mediapipe as mp
from time import time
from pynput.keyboard import Controller, Key
import threading

class VirtualKeyboard:
    def __init__(self):
        # Thiet lap camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)  # chieu rong
        self.cap.set(4, 720)   # chieu cao

        # Thiet lap MediaPipe Hands voi cac thong so toi uu hon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.7,  # Giam nguong de de nhan dien hon
            min_tracking_confidence=0.4    # Giam nguong de theo doi tot hon
        )
        self.mpDraw = mp.solutions.drawing_utils

        # Bo dieu khien ban phim
        self.keyboard = Controller()
        
        # Bo tri ban phim voi phim dac biet
        self.keys = [
            list("QWERTYUIOP"),
            list("ASDFGHJKL;"),
            list("ZXCVBNM,./")
        ]
        
        # Phim dac biet
        self.special_keys = [
            {"text": "SPACE", "key": Key.space, "size": (200, 85), "pos": (400, 450)},
            {"text": "BACK", "key": Key.backspace, "size": (150, 85), "pos": (650, 450)},
            {"text": "ENTER", "key": Key.enter, "size": (150, 85), "pos": (850, 450)}
        ]
        
        # Tao danh sach nut
        self.buttonList = self.create_buttons()
        
        # Bien van ban va thoi gian
        self.finalText = ""
        self.last_click = 0
        self.click_delay = 0.2  # Giam thoi gian tre de phan hoi nhanh hon
        
        # Theo doi trang thai ngon tay
        self.clicked = False
        self.hover_time = 0    # Thoi gian hover tren nut
        self.current_hover_btn = None  # Nut dang duoc hover
        
        # Hang so mau sac
        self.BUTTON_BG = (100, 0, 140)     # Tim nhat
        self.BUTTON_HOVER = (140, 0, 200)  # Tim sang hon
        self.BUTTON_TEXT = (255, 255, 255) # Trang
        self.TEXT_BOX_BG = (50, 0, 90)     # Tim dam
        self.CLICK_COLOR = (0, 255, 0)     # Mau khi click

        # Theo doi hieu suat
        self.frame_times = []
        self.last_frame_time = time()
        self.show_fps = True
        
        # Che do debug
        self.debug_mode = True  # Hien thi thong tin debug

    def create_buttons(self):
        buttons = []
        # Phim chuan
        for i, row in enumerate(self.keys):
            for j, key in enumerate(row):
                buttons.append(Button((100*j + 50, 100*i + 100), key))
        
        # Phim dac biet
        for special in self.special_keys:
            buttons.append(Button(special["pos"], special["text"], special["size"]))
            
        return buttons

    def draw_keyboard(self, img):
        # Ve tat ca cac nut
        for btn in self.buttonList:
            x, y = btn.pos
            w, h = btn.size
            # Nen nut
            cv2.rectangle(img, (x, y), (x+w, y+h), self.BUTTON_BG, cv2.FILLED)
            # Vien nut
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 2)
            # Van ban nut
            if len(btn.text) > 1:  # Phim dac biet co van ban nho hon
                cv2.putText(img, btn.text, (x+10, y+50), 
                            cv2.FONT_HERSHEY_PLAIN, 2, self.BUTTON_TEXT, 2)
            else:  # Phim thuong
                cv2.putText(img, btn.text, (x+25, y+60),
                            cv2.FONT_HERSHEY_PLAIN, 4, self.BUTTON_TEXT, 4)
        return img

    def draw_text_box(self, img):
        # Ve hop van ban
        cv2.rectangle(img, (50, 500), (1000, 600), self.TEXT_BOX_BG, cv2.FILLED)
        cv2.rectangle(img, (50, 500), (1000, 600), (255, 255, 255), 2)
        
        # Hien thi van ban voi con tro
        displayed_text = self.finalText[-40:] if len(self.finalText) > 40 else self.finalText
        displayed_text_with_cursor = displayed_text + '|'  # Them con tro
        cv2.putText(img, displayed_text_with_cursor, (60, 570),
                    cv2.FONT_HERSHEY_PLAIN, 3, self.BUTTON_TEXT, 3)
        
        # Them huong dan
        cv2.putText(img, "Van ban se xuat hien o day va trong ung dung dang duoc chon", 
                    (50, 480), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
        
        return img

    def process_hand(self, img):
        lmList = []
        # Them thong bao trang thai nhan dien
        status_text = "Dang tim ban tay..."
        status_color = (0, 0, 255)  # Do khi chua tim thay ban tay
        
        # Chuyen doi hinh anh sang RGB de MediaPipe xu ly
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        
        if results.multi_hand_landmarks:
            # Da tim thay ban tay
            handLms = results.multi_hand_landmarks[0]
            status_text = "Da phat hien ban tay"
            status_color = (0, 255, 0)  # Xanh la khi tim thay
            
            # Lay tat ca cac diem moc
            for id, lm in enumerate(handLms.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((cx, cy))
            
            # Ve cac diem moc ban tay voi hieu ung noi bat
            self.mpDraw.draw_landmarks(
                img, handLms, self.mpHands.HAND_CONNECTIONS,
                self.mpDraw.DrawingSpec(color=(0, 255, 255), thickness=3, circle_radius=3),  # Diem moc mau vang
                self.mpDraw.DrawingSpec(color=(0, 0, 255), thickness=2)  # Duong ket noi mau do
            )
            
            # Danh dau dac biet cho cac diem quan trong
            if len(lmList) >= 21:
                # Dau ngon tro
                cv2.circle(img, lmList[8], 12, (0, 255, 0), cv2.FILLED)
                # Dau ngon giua
                cv2.circle(img, lmList[12], 12, (0, 0, 255), cv2.FILLED)
        
        # Hien thi trang thai nhan dien
        cv2.putText(img, status_text, (50, 680), 
                    cv2.FONT_HERSHEY_PLAIN, 2, status_color, 2)
        
        return lmList, img

    def handle_button_interaction(self, img, lmList):
        if not lmList or len(lmList) < 21:  # Dam bao da nhan du cac diem landmarks
            return img
            
        # Lay vi tri dau ngon tay
        index_finger_tip = lmList[8]      # Dau ngon tro
        middle_finger_tip = lmList[12]    # Dau ngon giua
        index_finger_pip = lmList[6]      # Khop giua ngon tro
        
        # Kiem tra xem ngon tro co dang gio len khong (de chuan bi nhan)
        index_finger_up = index_finger_tip[1] < index_finger_pip[1]
        
        # Ve vong tron o dau ngon tro de phan hoi truc quan tot hon
        cv2.circle(img, index_finger_tip, 15, (0, 255, 0), cv2.FILLED)
        
        # Hien thi thong tin debug ve khoang cach giua cac ngon tay
        dist = np.hypot(middle_finger_tip[0] - index_finger_tip[0], 
                        middle_finger_tip[1] - index_finger_tip[1])
        if self.debug_mode:
            cv2.putText(img, f"Dist: {int(dist)}", (950, 50), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        
        # Kiem tra hover va click cho tung nut
        for btn in self.buttonList:
            x, y = btn.pos
            w, h = btn.size
            
            # Kiem tra xem ngon tro co dang o tren nut khong
            if x < index_finger_tip[0] < x+w and y < index_finger_tip[1] < y+h:
                # To sang nut khi hover
                cv2.rectangle(img, (x-5, y-5), (x+w+5, y+h+5), self.BUTTON_HOVER, cv2.FILLED)
                
                # Ve chu len nut da to sang
                if len(btn.text) > 1:  # Phim dac biet
                    cv2.putText(img, btn.text, (x+10, y+50), 
                                cv2.FONT_HERSHEY_PLAIN, 2, self.BUTTON_TEXT, 2)
                else:  # Phim thuong
                    cv2.putText(img, btn.text, (x+25, y+60),
                                cv2.FONT_HERSHEY_PLAIN, 4, self.BUTTON_TEXT, 4)
                
                # Cai thien: Hai phuong phap phat hien click
                clicked = False
                
                # Phuong phap 1: Dua vao khoang cach giua ngon tro va ngon giua
                if dist < 40:  # Tang nguong de de nhan dien hon
                    clicked = True
                
                # Phuong phap 2: Su dung thoi gian ngon tay o tren nut
                if index_finger_up and time() - self.last_click > self.click_delay:
                    # Danh dau khu vuc hover de nguoi dung de nhin
                    cv2.circle(img, index_finger_tip, 20, (0, 0, 255), 4)
                
                if clicked and time() - self.last_click > self.click_delay:
                    self.last_click = time()
                    
                    # Xu ly nhan phim dua tren nut
                    self.handle_key_press(btn)
                    
                    # Phan hoi truc quan cho click
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), cv2.FILLED)
                    if len(btn.text) > 1:
                        cv2.putText(img, btn.text, (x+10, y+50), 
                                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
                    else:
                        cv2.putText(img, btn.text, (x+25, y+60),
                                    cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)
                    
                    # Phan hoi am thanh (neu co the)
                    print(f"Da nhan: {btn.text}")
                    
                    # Them hinh hieu ung click
                    cv2.circle(img, index_finger_tip, 30, (0, 255, 255), cv2.FILLED)
        
        return img

    def handle_key_press(self, btn):
        # Su dung threading de tranh do tre giao dien khi nhan phim
        def press_key():
            try:
                text = btn.text
                # Xu ly cac phim dac biet
                if text == "SPACE":
                    self.keyboard.press(Key.space)
                    self.keyboard.release(Key.space)
                    self.finalText += " "
                elif text == "BACK":
                    if len(self.finalText) > 0:
                        self.keyboard.press(Key.backspace)
                        self.keyboard.release(Key.backspace)
                        self.finalText = self.finalText[:-1]
                elif text == "ENTER":
                    self.keyboard.press(Key.enter)
                    self.keyboard.release(Key.enter)
                    self.finalText += "\n"
                else:  # Phim thuong
                    # Xu ly theo tung ky tu
                    for char in text:
                        self.keyboard.press(char.lower())
                        self.keyboard.release(char.lower())
                    self.finalText += text
                
                # In thong bao ra console de xac nhan
                print(f"Da nhan phim: {text}")
            except Exception as e:
                print(f"Loi khi nhan phim {btn.text}: {str(e)}")
        
        # Bat dau luong nhan phim rieng biet
        thread = threading.Thread(target=press_key)
        thread.daemon = True  # Dam bao luong ket thuc khi chuong trinh chinh ket thuc
        thread.start()

    def calculate_fps(self):
        current_time = time()
        fps = 1 / (current_time - self.last_frame_time)
        self.last_frame_time = current_time
        
        # Giu chi 30 khung hinh gan nhat de tinh trung binh
        self.frame_times.append(fps)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        
        return int(sum(self.frame_times) / len(self.frame_times))

    def display_fps(self, img, fps):
        cv2.putText(img, f"FPS: {fps}", (50, 50), 
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        return img

    def run(self):
        try:
            print("Bat dau ban phim ao. Nhan 'q' de thoat.")
            print("Huong dan su dung:")
            print("1. Di chuyen ngon tro de di chuyen con tro tren ban phim")
            print("2. Dua ngon tro va ngon giua lai gan nhau de nhan phim")
            print("3. Cac phim dac biet: SPACE, BACK, ENTER")
            
            while True:
                # Doc khung hinh
                success, img = self.cap.read()
                if not success:
                    print("Khong the doc khung hinh tu camera")
                    # Thu ket noi lai camera neu mat ket noi
                    self.cap = cv2.VideoCapture(0)
                    continue
                    
                # Lat hinh anh theo chieu ngang de trai nghiem tot hon
                img = cv2.flip(img, 1)
                
                # Tao ban sao de tranh thay doi anh goc
                display_img = img.copy()
                
                # Xu ly diem moc ban tay
                lmList, display_img = self.process_hand(display_img)
                
                # Ve ban phim
                display_img = self.draw_keyboard(display_img)
                
                # Xu ly tuong tac nut
                display_img = self.handle_button_interaction(display_img, lmList)
                
                # Ve hop van ban
                display_img = self.draw_text_box(display_img)
                
                # Tinh toan va hien thi FPS
                if self.show_fps:
                    fps = self.calculate_fps()
                    display_img = self.display_fps(display_img, fps)
                
                # Hien thi huong dan
                cv2.putText(display_img, "Di chuyen ngon tro de chon phim, ep ngon tro & giua de click", 
                            (50, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
                
                # Hien thi hinh anh
                cv2.imshow("Ban Phim Ao Nang Cao", display_img)
                
                # Kiem tra phim thoat
                key = cv2.waitKey(1) 
                if key & 0xFF == ord('q'):
                    print("Dang thoat chuong trinh...")
                    break
                elif key & 0xFF == ord('d'):
                    # Bat/tat che do debug
                    self.debug_mode = not self.debug_mode
                    print(f"Che do debug: {'Bat' if self.debug_mode else 'Tat'}")
                    
        except Exception as e:
            print(f"Loi: {str(e)}")
        finally:
            # Giai phong tai nguyen
            self.cap.release()
            cv2.destroyAllWindows()
            print("Da dong ban phim ao.")


class Button:
    def __init__(self, pos, text, size=(85, 85)):
        self.pos = pos
        self.size = size
        self.text = text


if __name__ == "__main__":
    keyboard = VirtualKeyboard()
    keyboard.run()