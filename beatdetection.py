import librosa
import pygame
import numpy as np
import sys
import threading
import time

# Initialize pygame
pygame.init()

# Screen settings
screenWidth = 800
screenHeight = 600
screen = pygame.display.set_mode((screenWidth, screenHeight))
pygame.display.set_caption("Beat Detection App")

# Colors
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

# Dot settings
dotX = screenWidth // 2
dotY = screenHeight // 2
dotRadius = 50
beatDetected = False
beatFlashTime = 0
flashDuration = 0.1  # Flash duration in seconds

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
    global beatDetected, beatFlashTime
    
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
                beatFlashTime = time.time()  # Record when the flash started
                beatIndex += 1
                print(f"Beat detected at {currentTime:.2f} seconds!")
            
            time.sleep(0.01)  # Small delay to prevent excessive CPU usage
            
    except Exception as e:
        print(f"Error playing audio: {e}")

def drawScreen():
    global beatDetected, beatFlashTime
    
    # Clear screen
    screen.fill(BLACK)
    
    currentTime = time.time()
    if beatDetected and (currentTime - beatFlashTime) > flashDuration:
        beatDetected = False  # Stop flashing after duration
    if beatDetected:
        pygame.draw.circle(screen, GREEN, (dotX, dotY), dotRadius)
    else:
        pygame.draw.circle(screen, WHITE, (dotX, dotY), dotRadius)
    
    # Add instructions
    font = pygame.font.Font(None, 36)
    text = font.render("Beat Detection - Green dot flashes on beats", True, WHITE)
    textRect = text.get_rect(center=(screenWidth // 2, 100))
    screen.blit(text, textRect)
    
    # Update display
    pygame.display.flip()

def main():
    global audioFile
    
    print("Beat Detection App")
    print("=" * 20)
    
    # Check if audio file exists or get from user
    if len(sys.argv) > 1:
        audioFile = sys.argv[1]
    else:
        audioFile = input("Enter the path to your audio file: ")
    
    # Check if file exists
    import os
    if not os.path.exists(audioFile):
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
    
    print("Starting beat detection app...")
    print("Press ESC or close window to exit")
    
    audioThread = threading.Thread(target=playAudioWithBeats, args=(beatTimes,))
    audioThread.daemon = True
    audioThread.start()
    
    # Main game loop
    clock = pygame.time.Clock()
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        drawScreen()
        
        clock.tick(60)  # 60 FPS
    
    pygame.mixer.music.stop()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()