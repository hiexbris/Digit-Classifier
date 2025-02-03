import pygame
import torch
import numpy as np

# Constants
WIDTH, HEIGHT = 560, 610  # Added 100px for info panel
ROWS, COLS = 28, 28
CELL_SIZE = WIDTH // COLS

# Load weights
weights = torch.load('weights.pth')
W1 = weights['W1']
W2 = weights['W2']
W3 = weights['W3']
B1 = weights['B1']
B2 = weights['B2']
B3 = weights['B3']

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PANEL_COLOR = (230, 230, 230)

# Initialize grid and prediction variables
grid = [[0 for _ in range(COLS)] for _ in range(ROWS)]
predicted_number = 0
prediction_prob = 0.0

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("MNIST Digit Classifier")
font = pygame.font.Font(None, 36)
clock = pygame.time.Clock()

def softmax(Z3):
    exp_Z3 = np.exp(Z3 - np.max(Z3, axis=0))  
    return exp_Z3 / (np.sum(exp_Z3, axis=0, keepdims=True) + 1e-8)

def draw_grid():
    # Draw main grid
    for row in range(ROWS):
        for col in range(COLS):
            color = (grid[row][col], grid[row][col], grid[row][col])
            pygame.draw.rect(screen, color, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE - 1, CELL_SIZE - 1))
    
    # Draw info panel
    pygame.draw.rect(screen, PANEL_COLOR, (0, HEIGHT-50, WIDTH, 50))
    
    # Display prediction
    prediction_text = font.render(f"Prediction: {predicted_number} ({prediction_prob:.2%})", 
                                 True, BLACK)
    screen.blit(prediction_text, (20, HEIGHT-40))
    
    # Draw grid lines on top of colors
    for row in range(ROWS):
        for col in range(COLS):
            pygame.draw.rect(screen, BLACK, (col * CELL_SIZE, row * CELL_SIZE, 
                                            CELL_SIZE - 1, CELL_SIZE - 1), 1)

def update_prediction():
    global predicted_number, prediction_prob
    # Convert grid to input format
    S0 = np.array(grid, dtype=np.float32).reshape(784, 1) / 255.0
    
    # Forward pass
    Z1 = np.dot(W1, S0) + B1
    S1 = np.maximum(0, Z1)
    
    Z2 = np.dot(W2, S1) + B2
    S2 = np.maximum(0, Z2)
    
    Z3 = np.dot(W3, S2) + B3
    S3 = softmax(Z3)
    
    # Get prediction
    predicted_number = np.argmax(S3)
    prediction_prob = S3[predicted_number][0]

def handle_mouse_input():
    if pygame.mouse.get_pressed()[0]:
        x, y = pygame.mouse.get_pos()
        col, row = x // CELL_SIZE, y // CELL_SIZE
        if 0 <= row < ROWS and 0 <= col < COLS:
            # Add drawing with smooth effect
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < ROWS and 0 <= nc < COLS:
                        grid[nr][nc] = min(255, grid[nr][nc] + 100)
            update_prediction()

def main():
    global predicted_number, prediction_prob
    running = True
    while running:
        screen.fill(WHITE)
        handle_mouse_input()
        draw_grid()
        pygame.display.flip()
        clock.tick(60)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    for row in range(ROWS):
                        for col in range(COLS):
                            grid[row][col] = 0
                    predicted_number = 0
                    prediction_prob = 0.0

if __name__ == "__main__":
    main()