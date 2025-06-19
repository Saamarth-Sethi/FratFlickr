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

# Audio settings
audioFile = "your_song.mp3"  # Replace with your audio file path

def loadAndAnalyzeAudio(filename):
    """Load audio file and detect beats"""
    try:
        print("Loading audio file...")
        # Load audio file with explicit duration limit to avoid memory issues
        y, sr = librosa.load(filename, duration=None)
        print(f"Audio loaded successfully. Sample rate: {sr}, Duration: {len(y)/sr:.2f} seconds")
        
        print("Analyzing beats...")
        # Detect beat times with more robust parameters
        tempo, beatFrames = librosa.beat.beat_track(y=y, sr=sr, hop_length=512, start_bpm=120.0)
        beatTimes = librosa.frames_to_time(beatFrames, sr=sr, hop_length=512)
        
        print(f"Detected tempo: {float(tempo):.2f} BPM")
        print(f"Found {len(beatTimes)} beats")
        
        return beatTimes, float(tempo)
        
    except FileNotFoundError:
        print(f"Error: Could not find file '{filename}'")
        print("Please check that the file path is correct and the file exists.")
        return None, None
    except Exception as e:
        print(f"Error loading audio: {str(e)}")
        print("This might be due to:")
        print("1. Unsupported audio format")
        print("2. Corrupted audio file") 
        print("3. Library compatibility issues")
        print("Try converting your file to WAV format or updating librosa/numpy")
        return None, None

def playAudioWithBeats(beatTimes):
    """Play audio and trigger beat detection"""
    global beatDetected
    
    # Initialize pygame mixer for audio playback
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
                beatIndex += 1
                print(f"Beat detected at {currentTime:.2f} seconds!")
            
            time.sleep(0.01)  # Small delay to prevent excessive CPU usage
            
    except Exception as e:
        print(f"Error playing audio: {e}")

def drawScreen():
    """Draw the screen with the beat indicator dot"""
    global beatDetected
    
    # Clear screen
    screen.fill(BLACK)
    
    # Draw dot - green if beat detected, white otherwise
    if beatDetected:
        pygame.draw.circle(screen, GREEN, (dotX, dotY), dotRadius)
        # Reset beat detection after drawing
        beatDetected = False
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
    """Main function to run the beat detection app"""
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
        # print(f"Error: File '{audioFile}' not found!")
        return
    
    try:
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning)
    except:
        pass
    
    print("Loading and analyzing audio...")
    beatTimes, tempo = loadAndAnalyzeAudio(audioFile)
    
    if beatTimes is None:
        print("\nTroubleshooting tips:")
        print("1. Try converting your file to WAV format")
        print("2. Update libraries: pip install --upgrade librosa numpy")
        print("3. Make sure the file path has no special characters")
        return
    
    print("Starting beat detection app...")
    print("Press ESC or close window to exit")
    
    # Start audio playback in a separate thread
    audioThread = threading.Thread(target=playAudioWithBeats, args=(beatTimes,))
    audioThread.daemon = True
    audioThread.start()
    
    # Main game loop
    clock = pygame.time.Clock()
    running = True
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Draw everything
        drawScreen()
        
        # Control frame rate
        clock.tick(60)  # 60 FPS
    
    # Cleanup
    pygame.mixer.music.stop()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()