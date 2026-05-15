import numpy as np

def polar_to_cartesian(angle_deg, distance_mm):

    theta = np.radians(angle_deg)

    x = distance_mm * np.cos(theta)
    y = distance_mm * np.sin(theta)

    return x, y


# Example usage
if __name__ == "__main__":

    angle = 45
    distance = 1000  # mm

    x, y = polar_to_cartesian(angle, distance)

    print(f"X: {x:.2f} mm")
    print(f"Y: {y:.2f} mm")