from manim import *
import numpy as np
import os

def get_parameterized_2_data(num_points=50, noise_level=0.10):
    """Generates the 3D point cloud for the parameterized '2'."""
    np.random.seed(42)  # For reproducibility
    t = np.linspace(0, 3.0, num_points)
    x = np.zeros_like(t)
    y = np.zeros_like(t)

    top_arc_end = np.argwhere(t <= 1.0)[-1].item()
    middle_arc_end = np.argwhere(t <= 2.0)[-1].item()

    x[:top_arc_end] = -np.cos(np.pi * t[:top_arc_end])
    x[top_arc_end:middle_arc_end] = 3 - 2 * t[top_arc_end:middle_arc_end]
    x[middle_arc_end:] = 2 * t[middle_arc_end:] - 5

    y[:top_arc_end] = 0.5 + np.sin(np.pi * t[:top_arc_end])
    y[top_arc_end:middle_arc_end] = 2 - 1.5 * t[top_arc_end:middle_arc_end]
    y[middle_arc_end:] = -1.0

    x_noisy = x + np.random.normal(scale=noise_level, size=x.shape)
    y_noisy = y + np.random.normal(scale=noise_level, size=y.shape)
    z_noisy = np.random.normal(scale=noise_level * 3, size=x.shape)

    return np.vstack((x_noisy, y_noisy, z_noisy)).T

def get_pca_basis(points_3d):
    """Calculates the first two PCA basis vectors."""
    mean = np.mean(points_3d, axis=0)
    centered_data = points_3d - mean
    cov_matrix = np.cov(centered_data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    return sorted_eigenvectors[:, 0], sorted_eigenvectors[:, 1]

class PCAProjection(ThreeDScene):
    def construct(self):
        # --- Step 1: Scene Setup and Data Generation ---
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        # Generate the 3D points
        points_3d = get_parameterized_2_data()
        dots = VGroup(*[Sphere(point, radius=0.05, color=RED) for point in points_3d])
        self.add(dots)
        self.begin_ambient_camera_rotation()
        self.wait(3)
        pc1, pc2 = get_pca_basis(points_3d)
        data_mean = np.mean(points_3d, axis=0)

        # --- Step 2: Compute and Visualize PCA Basis Vectors ---
        pc1_arrow = Vector(direction=data_mean + pc1 * 2, color="#FFD700").set_sheen(0.5, direction=pc1)
        pc2_arrow = Vector(direction=data_mean + pc2 * 2, color="#C0C0C0").set_sheen(0.5, direction=pc2)
        #  pc1_label = Text("PC1", font_size=24).next_to(pc1_arrow.get_end(), RIGHT)
        #  pc2_label = Text("PC2", font_size=24).next_to(pc2_arrow.get_end(), RIGHT)

        # self.play(Create(pc1_arrow), Write(pc1_label))
        # self.play(Create(pc2_arrow), Write(pc2_label))
        self.play(Create(pc1_arrow))
        self.play(Create(pc2_arrow))
        self.wait(1)

        # --- Step 3: Create and Animate the PCA Plane ---
        plane = Surface(
            lambda u, v: data_mean + u * pc1 + v * pc2,
            u_range=[-2.5, 2.5],
            v_range=[-2.5, 2.5],
            resolution=(4, 4),
            fill_opacity=0.2
        )
        plane.set_style(fill_color=GREY, stroke_color=GREY)

        self.play(Create(plane), run_time=2)
        self.wait(1)

        # --- Step 4: Animate the Projection ---
        # Calculate projected points
        centered_points = points_3d - data_mean
        coords_in_pca_plane = centered_points @ np.vstack((pc1, pc2)).T
        projected_points_centered = coords_in_pca_plane @ np.vstack((pc1, pc2))
        projected_points = projected_points_centered + data_mean  # we don't want it to project onto the normalized plane -- we want the original plane for beauty

        # Create mobjects for projected points and projection lines
        projection_lines = VGroup(*[DashedLine(p_orig, p_proj) for p_orig, p_proj in zip(points_3d, projected_points)])
        projected_dots = VGroup(*[Dot(point, color=BLUE, radius=0.05) for point in projected_points])

        self.play(
            LaggedStart(
                *[Create(line) for line in projection_lines],
                lag_ratio=0.05,
                run_time=3
            )
        )
        self.play(Transform(dots, projected_dots), run_time=2)
        self.play(FadeOut(projection_lines))

        # --- Step 5: Final Touches and Camera Work ---
        self.move_camera(phi=0 * DEGREES, theta=-45 * DEGREES, gamma=45, run_time=3) #need to rotate 45 clockwise to see the plane better
        self.wait(4)

if __name__ == '__main__':
    scene = PCAProjection()
    scene.render()
    os.startfile(scene.renderer.file_writer.movie_file_path)