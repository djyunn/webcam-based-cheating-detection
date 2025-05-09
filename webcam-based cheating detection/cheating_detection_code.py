import cv2
import time
import numpy as np
import dlib
import sys
import os
from ultralytics import YOLO

# GazeTracking GitHub 저장소 경로 추가
# 클론한 저장소 위치에 맞게 경로 조정하세요
gaze_tracking_path = "GazeTracking"
sys.path.append(gaze_tracking_path)

try:
    from gaze_tracking import GazeTracking
except ImportError:
    print("GazeTracking 모듈을 찾을 수 없습니다.")
    print("1. https://github.com/antoinelame/GazeTracking을 클론하세요.")
    print("2. 클론한 디렉토리 경로를 gaze_tracking_path 변수에 설정하세요.")
    sys.exit(1)

class CheatingDetector:
    def __init__(self):
        # 웹캠 설정
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # 시선 추적 초기화
        self.gaze = GazeTracking()
        
        # YOLO 모델 로드
        try:
            self.yolo_model = YOLO("yolov8n.pt")  # 작은 모델 사용
            self.yolo_enabled = True
        except Exception as e:
            print(f"YOLO 모델 로드 실패: {e}")
            print("객체 탐지 기능 없이 계속 진행합니다.")
            self.yolo_enabled = False
        
        # 부정행위 관련 객체 클래스 ID (COCO 데이터셋 기준)
        self.cheating_objects = {
            67: "휴대폰",  # 'cell phone'
            73: "책"       # 'book'
        }
        
        # Dlib 얼굴 감지기 및 랜드마크 예측기 초기화
        try:
            self.detector = dlib.get_frontal_face_detector()
            landmark_path = os.path.join(gaze_tracking_path, "trained_models", "shape_predictor_68_face_landmarks.dat")
            if not os.path.exists(landmark_path):
                print(f"랜드마크 모델을 찾을 수 없습니다: {landmark_path}")
                print("GazeTracking 저장소 내의 trained_models 폴더에 shape_predictor_68_face_landmarks.dat 파일이 있는지 확인하세요.")
                raise FileNotFoundError(f"랜드마크 파일 없음: {landmark_path}")
            
            self.predictor = dlib.shape_predictor(landmark_path)
            self.head_pose_enabled = True
        except Exception as e:
            print(f"머리 방향 추정 초기화 실패: {e}")
            print("머리 방향 추정 기능 없이 계속 진행합니다.")
            self.head_pose_enabled = False
        
        # 3D 모델 포인트 (머리 방향 추정용)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # 코 끝
            (0.0, -330.0, -65.0),        # 턱
            (-225.0, 170.0, -135.0),     # 왼쪽 눈 왼쪽 구석
            (225.0, 170.0, -135.0),      # 오른쪽 눈 오른쪽 구석
            (-150.0, -150.0, -125.0),    # 왼쪽 입 구석
            (150.0, -150.0, -125.0)      # 오른쪽 입 구석
        ])
        
        # 카메라 매트릭스
        self.camera_matrix = np.array(
            [[650, 0, 320],
             [0, 650, 240],
             [0, 0, 1]], dtype=np.float64
        )
        
        # 카메라 왜곡 계수
        self.dist_coeffs = np.zeros((4, 1))
        
        # 부정행위 카운터 초기화
        self.cheating_counter = 0
        self.last_warning_time = 0
        
        # 화면 ROI (관심 영역) 정의
        self.roi_left = 100
        self.roi_top = 100
        self.roi_right = 540
        self.roi_bottom = 380
    
    def get_head_pose(self, shape):
        """Dlib 랜드마크에서 머리 방향 추정"""
        # 랜드마크에서 특정 포인트 추출
        image_points = np.array([
            (shape.part(30).x, shape.part(30).y),     # 코 끝
            (shape.part(8).x, shape.part(8).y),       # 턱
            (shape.part(36).x, shape.part(36).y),     # 왼쪽 눈 왼쪽 구석
            (shape.part(45).x, shape.part(45).y),     # 오른쪽 눈 오른쪽 구석
            (shape.part(48).x, shape.part(48).y),     # 왼쪽 입 구석
            (shape.part(54).x, shape.part(54).y)      # 오른쪽 입 구석
        ], dtype=np.float64)
        
        # 회전 벡터와 변환 벡터 계산
        success, rotation_vec, translation_vec = cv2.solvePnP(
            self.model_points, image_points, self.camera_matrix, self.dist_coeffs
        )
        
        # 회전 매트릭스와 오일러 각도
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pose_mat = cv2.hconcat((rotation_mat, translation_vec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(
            cv2.hconcat((pose_mat, np.array([[0, 0, 0, 1]]).T))
        )
        
        return euler_angles
    
    def detect_cheating(self, frame):
        """부정행위 감지"""
        # 시선 추적 분석
        self.gaze.refresh(frame)
        frame = self.gaze.annotated_frame()
        
        is_cheating = False
        warning_text = ""
        
        # 1. 시선 추적 검사
        text = "시선 분석 중..."
        if self.gaze.is_blinking():
            text = "눈 깜빡임"
        elif self.gaze.is_right():
            text = "오른쪽 응시"
            self.cheating_counter += 1
        elif self.gaze.is_left():
            text = "왼쪽 응시"
            self.cheating_counter += 1
        elif self.gaze.is_center():
            text = "중앙 응시"
            self.cheating_counter = 0
        
        cv2.putText(frame, text, (20, 60), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        
        # 5초(150프레임) 이상 중앙 이외 응시 시 부정행위로 간주
        if self.cheating_counter >= 150:
            is_cheating = True
            warning_text += "시선이 화면 중앙을 벗어남! "
        
        # 2. YOLOv8 객체 감지 (활성화된 경우)
        if self.yolo_enabled:
            try:
                yolo_results = self.yolo_model(frame)
                for r in yolo_results:
                    boxes = r.boxes
                    for box in boxes:
                        cls_id = int(box.cls[0].item())
                        conf = box.conf[0].item()
                        
                        if cls_id in self.cheating_objects and conf > 0.5:
                            is_cheating = True
                            warning_text += f"{self.cheating_objects[cls_id]} 감지됨! "
                            
                            # 박스 그리기
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(frame, f"{self.cheating_objects[cls_id]}: {conf:.2f}", 
                                      (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            except Exception as e:
                cv2.putText(frame, "객체 탐지 오류", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
                print(f"YOLO 처리 중 오류: {e}")
        
        # 3. Dlib 얼굴 랜드마크 및 머리 방향 추정 (활성화된 경우)
        if self.head_pose_enabled:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.detector(gray)
                
                for face in faces:
                    # 랜드마크 감지
                    shape = self.predictor(gray, face)
                    
                    # 머리 방향 추정
                    euler_angles = self.get_head_pose(shape)
                    pitch, yaw, roll = euler_angles[0], euler_angles[1], euler_angles[2]
                    
                    # 머리 방향 정보 표시
                    cv2.putText(frame, f"Pitch: {pitch:.2f}", (20, 100), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1)
                    cv2.putText(frame, f"Yaw: {yaw:.2f}", (20, 130), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1)
                    cv2.putText(frame, f"Roll: {roll:.2f}", (20, 160), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1)
                    
                    # 머리 요(Yaw) 각도가 ±30도 이상이면 부정행위로 간주
                    if abs(yaw) > 30:
                        is_cheating = True
                        warning_text += "머리 방향이 화면을 벗어남! "
            except Exception as e:
                cv2.putText(frame, "머리 방향 추정 오류", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
                print(f"머리 방향 추정 중 오류: {e}")
        
        # 4. 화면 ROI 정의 및 표시
        cv2.rectangle(frame, (self.roi_left, self.roi_top), (self.roi_right, self.roi_bottom), (0, 255, 0), 2)
        
        # 부정행위 카운터 표시
        cv2.putText(frame, f"부정행위 카운터: {self.cheating_counter}/150", (400, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
        
        # 부정행위 감지 시 경고 표시
        if is_cheating:
            current_time = time.time()
            if current_time - self.last_warning_time > 3:  # 3초마다 콘솔 경고
                print("부정행위 가능성 감지!", warning_text)
                self.last_warning_time = current_time
                
            # 화면에 경고 메시지 표시
            cv2.putText(frame, "부정행위 감지!", (250, 70), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(frame, warning_text, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        
        return frame
    
    def run(self):
        """메인 실행 루프"""
        print("부정행위 탐지 시스템을 시작합니다...")
        print("종료하려면 'q' 키를 누르세요.")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("웹캠에서 프레임을 읽을 수 없습니다.")
                break
                
            # 프레임 미러링 (더 직관적인 사용자 경험 위해)
            frame = cv2.flip(frame, 1)
            
            # 부정행위 탐지
            try:
                frame = self.detect_cheating(frame)
            except Exception as e:
                print(f"프레임 처리 중 오류: {e}")
                # 오류 정보 표시
                cv2.putText(frame, f"처리 오류: {str(e)[:50]}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # 프레임 표시
            cv2.imshow("온라인 시험 부정행위 탐지", frame)
            
            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 리소스 해제
        self.cap.release()
        cv2.destroyAllWindows()
        print("부정행위 탐지 시스템이 종료되었습니다.")

if __name__ == "__main__":
    try:
        detector = CheatingDetector()
        detector.run()
    except Exception as e:
        print(f"오류 발생: {e}")
