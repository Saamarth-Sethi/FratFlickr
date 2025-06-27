import librosa
import pygame
import numpy as np
import sys
import threading
import time
import cv2
import mediapipe as mp
import os

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
        imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRgb)
        
        if self.results.multi_hand_landmarks:
            for handLandmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLandmarks, 
                                             self.mpHands.HAND_CONNECTIONS)
        return img
    
    def findPosition(self, img, handNumber=0, draw=True):
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

# Initialize pygame
pygame.init()

# Screen settings
screenWidth = 1200
screenHeight = 800
screen = pygame.display.set_mode((screenWidth, screenHeight))
pygame.display.set_caption("Beat Flick Challenge")

# Colors
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Game variables
score = 0
beatDetected = False
beatFlashTime = 0
flashDuration = 0.2  # Beat indicator duration
beatWindow = 0.3  # Time window around beat where flicks count for points
currentBeatTime = 0
lastFlickTime = 0

# Audio settings
audioFile = "your_song.mp3"

def loadAndAnalyzeAudio(filename):
    try:
        print("Loading audio file...")
        y, sr = librosa.load(filename, duration=None)
        print(f"Audio loaded successfully. Sample rate: {sr}, Duration: {len(y)/sr:.2f} seconds")
        
        print("Analyzing beats...")
        tempo, beatFrames = librosa.beat.beat_track(y=y, sr=sr, hop_length=512, start_bpm=120.0)
        beatTimes = librosa.frames_to_time(beatFrames, sr=sr, hop_length=512)
        
        print(f"Detected tempo: {float(tempo):.2f} BPM")
        print(f"Found {len(beatTimes)} beats")
        
        return beatTimes, float(tempo)
        
    except FileNotFoundError:
        print(f"Error: Could not find file '{filename}'")
        return None, None
    except Exception as e:
        print(f"Error loading audio: {str(e)}")
        return None, None

def playAudioWithBeats(beatTimes):
    global beatDetected, beatFlashTime, currentBeatTime
    
    pygame.mixer.init()
    
    try:
        # Load and play the audio file
        pygame.mixer.music.load(audioFile)
        pygame.mixer.music.play()
        
        # Start timing for beat detection
        startTime = time.time()
        beatIndex = 0
        
        while pygame.mixer.music.get_busy() and beatIndex < len(beatTimes):
            currentTime = time.time() - startTime
            
            # Check if we've reached the next beat time
            if beatIndex < len(beatTimes) and currentTime >= beatTimes[beatIndex]:
                beatDetected = True
                beatFlashTime = time.time()
                currentBeatTime = time.time()
                beatIndex += 1
                print(f"Beat detected at {currentTime:.2f} seconds!")
            
            time.sleep(0.01)  # Small delay to prevent excessive CPU usage
            
    except Exception as e:
        print(f"Error playing audio: {e}")

def checkFlickTiming():
    """Check if a flick occurred during the beat window"""
    global currentBeatTime, lastFlickTime, score
    
    if currentBeatTime > 0 and lastFlickTime > 0:
        timeDifference = abs(lastFlickTime - currentBeatTime)
        if timeDifference <= beatWindow:
            score += 1
            print(f"Perfect flick! Score: {score}")
            return True
    return False

def drawGameScreen(cameraImg=None):
    global beatDetected, beatFlashTime, score
    
    # Clear screen
    screen.fill(BLACK)
    
    # Draw camera feed if available
    if cameraImg is not None:
        # Convert OpenCV image to pygame surface (already mirrored in main loop)
        cameraImg = cv2.resize(cameraImg, (400, 300))
        cameraImg = cv2.cvtColor(cameraImg, cv2.COLOR_BGR2RGB)
        cameraImg = np.rot90(cameraImg)
        cameraImg = pygame.surfarray.make_surface(cameraImg)
        screen.blit(cameraImg, (50, 50))
    
    # Draw beat indicator circle
    currentTime = time.time()
    if beatDetected and (currentTime - beatFlashTime) > flashDuration:
        beatDetected = False  # Stop flashing after duration
    
    circleX = screenWidth - 200
    circleY = 200
    circleRadius = 80
    
    if beatDetected:
        pygame.draw.circle(screen, GREEN, (circleX, circleY), circleRadius)
        pygame.draw.circle(screen, WHITE, (circleX, circleY), circleRadius, 5)
    else:
        pygame.draw.circle(screen, WHITE, (circleX, circleY), circleRadius, 5)
    
    # Draw score
    font = pygame.font.Font(None, 72)
    scoreText = font.render(f"Score: {score}", True, YELLOW)
    scoreRect = scoreText.get_rect(center=(screenWidth // 2, 100))
    screen.blit(scoreText, scoreRect)
    
    # Draw instructions
    font = pygame.font.Font(None, 36)
    instructions = [
        "FratFlickr",
        "Flick your fingers on the beat when the circle turns GREEN!",
        "Press ESC to quit"
    ]
    
    for i, instruction in enumerate(instructions):
        text = font.render(instruction, True, WHITE)
        textRect = text.get_rect(center=(screenWidth // 2, 400 + i * 40))
        screen.blit(text, textRect)
    
    # Update display
    pygame.display.flip()

def main():
    global audioFile, lastFlickTime
    
    print("FratFlickr")
    print("=" * 30)
    
    # Check if audio file exists or get from user
    if len(sys.argv) > 1:
        audioFile = sys.argv[1]
    else:
        audioFile = input("Enter the path to your audio file: ")
    
    # Check if file exists
    if not os.path.exists(audioFile):
        print(f"File '{audioFile}' not found!")
        return
    
    try:
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning)
    except:
        pass
    
    print("Loading and analyzing audio...")
    beatTimes, tempo = loadAndAnalyzeAudio(audioFile)
    
    if beatTimes is None:
        return
    
    print("Starting FratFlickr...")
    print("Make sure your camera is connected!")
    print("Press ESC to exit")
    
    # Initialize camera and hand detector
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open camera!")
        return
    
    detector = HandDetector()
    
    # Start audio thread
    audioThread = threading.Thread(target=playAudioWithBeats, args=(beatTimes,))
    audioThread.daemon = True
    audioThread.start()
    
    # Main game loop
    clock = pygame.time.Clock()
    running = True
    
    while running:
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Get camera frame
        success, img = camera.read()
        if not success:
            print("Failed to read from camera")
            break
        
        # Don't flip the image - keep it natural
        
        # Find hands and detect flicks
        img = detector.findHands(img)
        landmarkList = detector.findPosition(img, draw=True)
        flickDetected = detector.detectFlick(landmarkList)
        
        if flickDetected:
            lastFlickTime = time.time()
            perfectTiming = checkFlickTiming()
            
            # Add visual feedback for flick
            cv2.putText(img, "FLICK!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                       2, (0, 255, 0) if perfectTiming else (0, 0, 255), 3)
        
        # Draw the game screen
        drawGameScreen(img)
        
        clock.tick(60)  # 60 FPS
    
    # Cleanup
    pygame.mixer.music.stop()
    camera.release()
    cv2.destroyAllWindows()
    pygame.quit()
    
    print(f"Game Over! Final Score: {score}")
    sys.exit()

if __name__ == "__main__":
    main()