import cv2
import mediapipe as mp
import time

class HandDetector:
    def __init__(self):
        # Initialize MediaPipe hands
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mpDraw = mp.solutions.drawing_utils
        
        # Store previous hand positions for motion detection
        self.previousPositions = []
        self.flickDetected = False
        self.flickCooldown = 0  # Prevent multiple detections for same flick
        self.cooldownFrames = 15  # Number of frames to wait before detecting next flick
        
    def findHands(self, img, draw=True):
        """Find hands in the image and optionally draw landmarks"""
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        
        if self.results.multi_hand_landmarks:
            for handLandmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLandmarks, 
                                             self.mpHands.HAND_CONNECTIONS)
        return img
    
    def findPosition(self, img, handNumber=0, draw=True):
        """Get the position of hand landmarks (focusing on finger tips)"""
        landmarkList = []
        
        if self.results.multi_hand_landmarks:
            if handNumber < len(self.results.multi_hand_landmarks):
                myHand = self.results.multi_hand_landmarks[handNumber]
                
                for id, landmark in enumerate(myHand.landmark):
                    height, width, channels = img.shape
                    centerX = int(landmark.x * width)
                    centerY = int(landmark.y * height)
                    landmarkList.append([id, centerX, centerY])
                    
                    if draw:
                        if id == 8:  # Index finger tip
                            cv2.circle(img, (centerX, centerY), 10, (255, 0, 255), cv2.FILLED)
                        elif id == 12:  # Middle finger tip
                            cv2.circle(img, (centerX, centerY), 10, (0, 255, 255), cv2.FILLED)
                        
        return landmarkList
    
    def detectFlick(self, landmarkList):
        """Detect flicking motion based on index and middle finger movement"""
        # Decrease cooldown counter
        if self.flickCooldown > 0:
            self.flickCooldown -= 1
            return False
            
        if len(landmarkList) == 0:
            return False
            
        # Get index finger tip (landmark 8) and middle finger tip (landmark 12)
        indexFingerTip = None
        middleFingerTip = None
        
        for landmark in landmarkList:
            if landmark[0] == 8:  # Index finger tip
                indexFingerTip = [landmark[1], landmark[2]]
            elif landmark[0] == 12:  # Middle finger tip
                middleFingerTip = [landmark[1], landmark[2]]
                
        # Need both fingers to be detected
        if indexFingerTip is None or middleFingerTip is None:
            return False
            
        # Calculate center point between the two finger tips
        centerX = (indexFingerTip[0] + middleFingerTip[0]) // 2
        centerY = (indexFingerTip[1] + middleFingerTip[1]) // 2
        fingerCenter = [centerX, centerY]
        
        # Store current position
        self.previousPositions.append(fingerCenter)
        
        # Keep only last 10 positions for analysis
        if len(self.previousPositions) > 10:
            self.previousPositions.pop(0)
            
        # Need at least 5 positions to detect motion
        if len(self.previousPositions) < 5:
            return False
            
        # Calculate velocity between consecutive positions
        velocities = []
        for i in range(1, len(self.previousPositions)):
            prevPos = self.previousPositions[i-1]
            currPos = self.previousPositions[i]
            
            # Calculate distance moved
            distance = ((currPos[0] - prevPos[0])**2 + (currPos[1] - prevPos[1])**2)**0.5
            velocities.append(distance)
            
        # Simple flick detection: look for fast movement
        if len(velocities) >= 4:
            # Get the most recent velocities
            recentVelocities = velocities[-3:]  # Last 3 velocities
            maxRecentVelocity = max(recentVelocities)
            
            # Flick detected if there's fast movement above threshold
            if maxRecentVelocity > 25:
                self.flickCooldown = self.cooldownFrames  # Set cooldown
                return True
                
        return False

def main():
    # Initialize camera and hand detector
    camera = cv2.VideoCapture(0)
    detector = HandDetector()
    
    # Counter for flicks
    flickCount = 0
    previousTime = 0
    
    #print("FratFlickr Hand Detection Started!")
    #print("Make flicking motions with your index and middle finger together!")
    #print("Press 'q' to quit")
    
    while True:
        success, img = camera.read()
        if not success:
            break
        img = cv2.flip(img, 1)
        
        # Find hands and get positions
        img = detector.findHands(img)
        landmarkList = detector.findPosition(img, draw=True)
        
        # Detect flick motion
        flickDetected = detector.detectFlick(landmarkList)
        
        if flickDetected:
            flickCount += 1
            # print(f"Flick detected! Total flicks: {flickCount}")
            
            cv2.putText(img, "FLICK!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                       2, (0, 255, 0), 3)
        
        # Calculate and display FPS
        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime
        
        # Display flick count and FPS
        cv2.putText(img, f"Flicks: {flickCount}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 255, 255), 2)
        cv2.putText(img, f"FPS: {int(fps)}", (10, img.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show the image
        cv2.imshow("FratFlickr - Hand Detection", img)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    camera.release()
    cv2.destroyAllWindows()
    # print(f"Hand detection ended! Total flicks: {flickCount}")

if __name__ == "__main__":
    main()