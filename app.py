import pygame
import torch
import numpy as np

# Constants
WIDTH, HEIGHT = 560, 660
ROWS, COLS = 28, 28
CELL_SIZE = WIDTH // COLS

# Color Palette
DARK_BLUE = (15, 15, 40)
DEEP_NAVY = (25, 25, 100)
PANEL_BLUE = (30, 30, 60)
BRIGHT_BLUE = (100, 100, 200)
WHITE = (230, 230, 255)

# Load weights
weights = torch.load('weights_0.1.pth')
W1 = weights['W1']
W2 = weights['W2']
W3 = weights['W3']
B1 = weights['B1']
B2 = weights['B2']
B3 = weights['B3']

# Initialize grid and prediction
grid = [[0.0 for _ in range(COLS)] for _ in range(ROWS)]
predicted_number = 0
prediction_prob = 0.0

# Pygame setup
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("MNIST Blue Canvas")
font = pygame.font.Font(None, 34)
clock = pygame.time.Clock()

def softmax(Z3):
    exp_Z3 = np.exp(Z3 - np.max(Z3, axis=0))  
    return exp_Z3 / (np.sum(exp_Z3, axis=0, keepdims=True) + 1e-8)

def remap(value):
    """Convert grid value (0-1) to blue color gradient"""
    intensity = int(value * 255)
    return (
        max(10, intensity // 4),
        max(10, intensity // 4),
        max(40, intensity)
    )

def draw_grid():
    # Draw main canvas
    for row in range(ROWS):
        for col in range(COLS):
            color = remap(grid[row][col])
            pygame.draw.rect(screen, color, 
                           (col*CELL_SIZE, row*CELL_SIZE, CELL_SIZE, CELL_SIZE))
    
    # Draw subtle grid lines
    for row in range(ROWS):
        for col in range(COLS):
            pygame.draw.rect(screen, DARK_BLUE, 
                           (col*CELL_SIZE, row*CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)
    
    # Draw info panel
    pygame.draw.rect(screen, PANEL_BLUE, (0, HEIGHT-100, WIDTH, 100))
    prediction_text = font.render(f"Predicted Digit: {predicted_number}", True, WHITE)
    confidence_text = font.render(f"Confidence: {prediction_prob*100:.1f}%", True, BRIGHT_BLUE)
    
    screen.blit(prediction_text, (20, HEIGHT-80))
    screen.blit(confidence_text, (20, HEIGHT-50))

def update_prediction():
    global predicted_number, prediction_prob
    S0 = np.array(grid).reshape(784, 1)
    
    # Network forward pass
    Z1 = np.dot(W1, S0) + B1
    S1 = np.maximum(0, Z1)
    Z2 = np.dot(W2, S1) + B2
    S2 = np.maximum(0, Z2)
    Z3 = np.dot(W3, S2) + B3
    S3 = softmax(Z3)
    
    predicted_number = np.argmax(S3)
    prediction_prob = S3[predicted_number][0]

def handle_mouse_input():
    x, y = pygame.mouse.get_pos()
    col, row = x // CELL_SIZE, y // CELL_SIZE
    
    if 0 <= row < ROWS and 0 <= col < COLS:
        # Precise drawing with pressure sensitivity
        if pygame.mouse.get_pressed()[0]:  # Left click
            grid[row][col] = min(1.0, grid[row][col] + 0.15)
        elif pygame.mouse.get_pressed()[2]:  # Right click to erase
            grid[row][col] = max(0.0, grid[row][col] - 0.1)

        # Subtle anti-aliasing
        for dr, dc in [(0,0), (0,1), (1,0), (0,-1), (-1,0)]:
            nr, nc = row+dr, col+dc
            if 0 <= nr < ROWS and 0 <= nc < COLS:
                if pygame.mouse.get_pressed()[0]:
                    grid[nr][nc] = min(1.0, grid[nr][nc] + 0.05)
                elif pygame.mouse.get_pressed()[2]:
                    grid[nr][nc] = max(0.0, grid[nr][nc] - 0.03)

def main():
    global predicted_number, prediction_prob
    running = True
    while running:
        screen.fill(DARK_BLUE)
        handle_mouse_input()
        update_prediction()
        draw_grid()
        pygame.display.flip()
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:  # Clear with spacebar
                    grid[:] = [[0.0]*COLS for _ in range(ROWS)]
                    prediction_prob = 0.0
                    predicted_number = 0

    pygame.quit()

if __name__ == "__main__":
    main()