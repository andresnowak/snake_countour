import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Load image
image = cv2.imread("coin.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)   # Grayscale for snake algorithm

# Display the image
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(image)
plt.title("Click to draw initial contour (enter or right-click to finish)")

# List to store clicked points
initial_points = []
lines = []  # Track all line segments
closing_line = None  # Track the line connecting last to first

# Mouse click event handler

def onclick(event):
    global initial_points, lines, closing_line

    # Left-click: add a point and draw lines
    if event.button == 1:
        x, y = event.xdata, event.ydata
        initial_points.append([y, x])  # (row, column) for scikit-image

        # Plot the point
        ax.plot(x, y, 'ro', markersize=5)

        # Draw lines between points
        if len(initial_points) > 1:
            prev_x, prev_y = initial_points[-2][1], initial_points[-2][0]
            line, = ax.plot([prev_x, x], [prev_y, y], 'r-', lw=2)
            lines.append(line)

        # Draw/update closing line to first point
        if closing_line is not None:
            closing_line.remove()  # Remove old closing line
        if len(initial_points) >= 2:
            first_x, first_y = initial_points[0][1], initial_points[0][0]
            closing_line, = ax.plot([x, first_x], [y, first_y], 'r--', lw=2)

        # Update the line dynamically
        plt.draw()

    # Right-click: close the contour (connect last to first)
    elif event.button == 3 and len(initial_points) >= 2:
        # Connect last point to first
        x_first, y_first = initial_points[0][1], initial_points[0][0]
        x_last, y_last = initial_points[-1][1], initial_points[-1][0]
        ax.plot([x_last, x_first], [y_last, y_first], 'r-', lw=2)
        plt.close()

# Right-click to finish drawing
def onkey(event):
    if event.key == 'enter':
        plt.close()

# Connect events to the figure
fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('key_press_event', onkey)
plt.show()

def get_blur_image_gradients(image, ksize=3, sigma=1):
    image_blur = cv2.GaussianBlur(image, (ksize, ksize), sigma, borderType=cv2.BORDER_CONSTANT)

    grad_x = cv2.Sobel(image_blur, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_CONSTANT)
    grad_y = cv2.Sobel(image_blur, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_CONSTANT)

    return grad_x, grad_y


def greedy_snake(image, initial_countour, alpha=0.3, beta=0.4, iterations=100, window_size=6, visualize=True):
    snake = initial_countour.copy() # the control points
    n_points = len(initial_countour)

    grad_x, grad_y = get_blur_image_gradients(image, ksize=3, sigma=5)

    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2) # gradient magnitude
    gradient_magnitude = cv2.GaussianBlur(gradient_magnitude, (15, 15), 5, borderType=cv2.BORDER_CONSTANT)  # Large blur

    # normalize gradients
    # gradient_magnitude = (gradient_magnitude - gradient_magnitude.min()) / (gradient_magnitude.max() - gradient_magnitude.min())
    edge_energy = -gradient_magnitude


    # search window space
    half_window = window_size//2

    offsets = np.array([(i, j) for i in range(-half_window, half_window + 1) for j in range(-half_window, half_window + 1)])

    if visualize:
        plt.figure(figsize=(10, 5))

    for iteration in range(iterations):
        new_snake = snake.copy()
        for i in range(n_points):
            x, y = snake[i]

            neighbors = np.round(snake[i] + offsets).astype(int)
            neighbors[:, 0] = np.clip(neighbors[:, 0], 0, image.shape[0] - 1)
            neighbors[:, 1] = np.clip(neighbors[:, 1], 0, image.shape[1] - 1)

            energies = []
            for nx, ny in neighbors:
                energy_image = edge_energy[nx, ny]

                prev_point = snake[(i - 1) % n_points]
                next_point = snake[(i + 1) % n_points]
                present_point = np.array([nx, ny])

                # the value of energy elastic and energy smooth is really big because we don't have a lot of points and the distance between the points is really large (so we need very small alpha and beta values)
                energy_elastic = np.sum((next_point - present_point)**2)
                energy_smooth = np.sum((next_point - 2 * present_point + prev_point)**2)

                total_energy = energy_image + alpha * energy_elastic + beta * energy_smooth

                energies.append(total_energy)

            # we want to minimize the sum of energies, so because each value is always positive  it is the same as just grabbing the smallest value each time.
            min_idx = np.argmin(energies)
            new_snake[i] = neighbors[min_idx]

        point_movement = np.mean(np.sqrt(np.sum((new_snake - snake)**2, axis=1)))
        if point_movement < 0.2:
            print(f"Converged at iteration {iteration} with average movement {point_movement:.2f}")
            break

        snake = new_snake

        if iteration % 20 == 0:
            snake = resample_snake(snake)

        # Visualize every N steps (e.g., every 5 iterations)
        if visualize and (iteration % 5 == 0 or iteration == iterations - 1):
            visualize_snake(initial_countour, snake, gradient_magnitude, iteration)

    if visualize:
        plt.show()

    return snake

def resample_snake(snake, n_points=None):
    if n_points is None:
        n_points = len(snake)

    # Close the snake for interpolation
    closed_snake = np.vstack([snake, snake[0]])

    # Calculate cumulative distance
    distances = np.cumsum(np.sqrt(np.sum(np.diff(closed_snake, axis=0)**2, axis=1)))
    distances = np.insert(distances, 0, 0)
    total_distance = distances[-1]

    # Interpolate
    interp_distances = np.linspace(0, total_distance, n_points, endpoint=False)
    fx = interp1d(distances, closed_snake[:, 0], kind='linear')
    fy = interp1d(distances, closed_snake[:, 1], kind='linear')

    new_x = fx(interp_distances)
    new_y = fy(interp_distances)

    return np.column_stack((new_x, new_y))

def visualize_snake(initial_countour, snake, gradient_norms, iteration):
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.plot(initial_countour[:, 1], initial_countour[:, 0], 'r--', label='Initial')
    plt.plot(snake[:, 1], snake[:, 0], 'b-', lw=2, label=f'Iter {iteration}')
    plt.title(f'Contour Evolution (Iteration {iteration})')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.imshow(gradient_norms, cmap='jet')
    plt.title('Gradient Magnitude (Energy)')
    plt.colorbar()

    plt.pause(0.1)  # Small delay to see updates

print("Image size", image.shape)

initial_points = np.array(initial_points)
final_contour = greedy_snake(gray_image, initial_points, alpha=0.003, beta=0.004, iterations=200, window_size=6)


plt.imshow(image, cmap='gray')
plt.plot(initial_points[:, 1], initial_points[:, 0], 'r--', label='Initial')
plt.plot(final_contour[:,1], final_contour[:,0], 'b-', lw=2, label='Final')
plt.legend()
plt.show()
