"""
Vector Geometry Visualization

Creates a 2D polar plot showing the geometric relationship between
prompted FK vector and steering vector.

Since we know cosine similarity, we can compute the angle between them
and plot both vectors to scale in the plane they span.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# Configuration
OUTPUT_DIR = Path("./output")
PROMPTED_VECTORS_PATH = OUTPUT_DIR / "prompted_fk_vectors.pt"
STEERING_VECTORS_PATH = OUTPUT_DIR / "complexity_vectors.pt"
BEST_LAYER = 6  # From previous analysis


def load_vectors():
    """Load both vector sets."""
    prompted = torch.load(PROMPTED_VECTORS_PATH)
    steering = torch.load(STEERING_VECTORS_PATH)
    return prompted[BEST_LAYER], steering[BEST_LAYER]


def compute_geometry(v1, v2):
    """
    Compute geometric relationship between two vectors.

    Returns:
        angle_rad: Angle between vectors in radians
        mag1: Magnitude of v1
        mag2: Magnitude of v2
        cos_sim: Cosine similarity
    """
    v1_np = v1.float().cpu().numpy()
    v2_np = v2.float().cpu().numpy()

    mag1 = np.linalg.norm(v1_np)
    mag2 = np.linalg.norm(v2_np)

    cos_sim = np.dot(v1_np, v2_np) / (mag1 * mag2)

    # Clamp to [-1, 1] to handle floating point errors
    cos_sim = np.clip(cos_sim, -1.0, 1.0)

    # Compute angle in radians
    angle_rad = np.arccos(cos_sim)

    return angle_rad, mag1, mag2, cos_sim


def create_polar_plot(angle_rad, mag1, mag2, cos_sim):
    """
    Create polar plot showing both vectors.

    Prompted vector at angle 0 (pointing right)
    Steering vector at computed angle
    """
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='polar')

    # Set up the plot
    ax.set_theta_zero_location('E')  # 0 degrees points right
    ax.set_theta_direction(1)  # Counter-clockwise

    # Plot prompted FK vector (angle 0, pointing right)
    ax.plot([0, 0], [0, mag1],
            color='#F18F01', linewidth=4,
            label=f'Prompted FK (mag={mag1:.2f})',
            marker='o', markersize=12, markevery=[1])

    # Plot steering vector (at computed angle)
    ax.plot([0, angle_rad], [0, mag2],
            color='#A23B72', linewidth=4,
            label=f'Steering (mag={mag2:.2f})',
            marker='s', markersize=12, markevery=[1])

    # Add arc showing angle between them
    if angle_rad > 0:
        arc_angles = np.linspace(0, angle_rad, 50)
        arc_radius = min(mag1, mag2) * 0.3  # 30% of smaller magnitude
        ax.plot(arc_angles, [arc_radius] * len(arc_angles),
                'g--', linewidth=2, alpha=0.5)

        # Add angle label
        mid_angle = angle_rad / 2
        label_radius = arc_radius * 1.3
        ax.text(mid_angle, label_radius,
                f'{np.degrees(angle_rad):.1f}°\n(cos={cos_sim:.3f})',
                ha='center', va='center',
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5',
                         facecolor='yellow', alpha=0.7))

    # Add magnitude comparison text
    mag_ratio = mag1 / mag2
    title_text = (
        f'Vector Geometry at Layer {BEST_LAYER}\n'
        f'Prompted FK vs Steering Vector\n\n'
        f'Magnitude Ratio: {mag_ratio:.3f} '
        f'(Prompted is {mag_ratio*100:.1f}% of Steering)'
    )
    ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20)

    # Style
    ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1.1), fontsize=12)
    ax.grid(True, alpha=0.3)

    # Set radial limits with some padding
    max_mag = max(mag1, mag2) * 1.2
    ax.set_ylim(0, max_mag)

    plt.tight_layout()

    return fig


def create_cartesian_plot(angle_rad, mag1, mag2, cos_sim):
    """
    Create Cartesian 2D plot in the plane spanned by the two vectors.

    Use prompted vector as x-axis, steering vector components in x and y.
    """
    fig, ax = plt.subplots(figsize=(12, 12))

    # Prompted vector points along x-axis
    prompted_x, prompted_y = mag1, 0

    # Steering vector at angle
    steering_x = mag2 * np.cos(angle_rad)
    steering_y = mag2 * np.sin(angle_rad)

    # Plot vectors as arrows from origin
    ax.arrow(0, 0, prompted_x, prompted_y,
             head_width=0.15, head_length=0.15,
             fc='#F18F01', ec='#F18F01', linewidth=3,
             length_includes_head=True, label='Prompted FK')

    ax.arrow(0, 0, steering_x, steering_y,
             head_width=0.15, head_length=0.15,
             fc='#A23B72', ec='#A23B72', linewidth=3,
             length_includes_head=True, label='Steering')

    # Add dashed line showing projection
    ax.plot([steering_x, steering_x], [0, steering_y],
            'g--', linewidth=2, alpha=0.5, label='Orthogonal component')

    # Add arc showing angle
    if angle_rad > 0:
        arc_angles = np.linspace(0, angle_rad, 50)
        arc_radius = min(mag1, mag2) * 0.3
        arc_x = arc_radius * np.cos(arc_angles)
        arc_y = arc_radius * np.sin(arc_angles)
        ax.plot(arc_x, arc_y, 'g--', linewidth=2, alpha=0.5)

        # Angle label
        mid_angle = angle_rad / 2
        label_x = arc_radius * 1.4 * np.cos(mid_angle)
        label_y = arc_radius * 1.4 * np.sin(mid_angle)
        ax.text(label_x, label_y,
                f'{np.degrees(angle_rad):.1f}°',
                ha='center', va='center',
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5',
                         facecolor='yellow', alpha=0.7))

    # Add component annotations
    ax.text(steering_x / 2, -0.3,
            f'Parallel component\n{steering_x:.2f} ({steering_x/mag2*100:.1f}% of steering)',
            ha='center', va='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))

    ax.text(steering_x + 0.3, steering_y / 2,
            f'Orthogonal component\n{steering_y:.2f} ({steering_y/mag2*100:.1f}% of steering)',
            ha='left', va='center', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))

    # Style
    ax.set_xlabel('Prompted FK direction', fontsize=12, fontweight='bold')
    ax.set_ylabel('Orthogonal direction', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Vector Decomposition at Layer {BEST_LAYER}\n'
        f'Cosine Similarity: {cos_sim:.4f}\n'
        f'Steering vector has {cos_sim*100:.1f}% alignment with Prompted direction',
        fontsize=14, fontweight='bold', pad=20
    )
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_aspect('equal')

    # Set limits with padding
    max_val = max(mag1, mag2) * 1.2
    ax.set_xlim(-0.5, max_val)
    ax.set_ylim(-0.5, max_val)

    plt.tight_layout()

    return fig


def main():
    """Create both visualizations."""
    print(f"\n{'='*80}")
    print("VECTOR GEOMETRY VISUALIZATION")
    print(f"{'='*80}\n")

    # Load vectors
    print(f"Loading vectors from layer {BEST_LAYER}...")
    prompted_vec, steering_vec = load_vectors()
    print("✓ Vectors loaded")

    # Compute geometry
    print("\nComputing geometric relationship...")
    angle_rad, mag1, mag2, cos_sim = compute_geometry(prompted_vec, steering_vec)

    print(f"\nResults:")
    print(f"  Prompted FK magnitude: {mag1:.4f}")
    print(f"  Steering magnitude: {mag2:.4f}")
    print(f"  Magnitude ratio: {mag1/mag2:.4f}")
    print(f"  Angle between vectors: {np.degrees(angle_rad):.2f}° ({angle_rad:.4f} rad)")
    print(f"  Cosine similarity: {cos_sim:.4f}")

    # Decompose steering into components
    parallel_component = mag2 * cos_sim  # Component along prompted direction
    orthogonal_component = mag2 * np.sin(angle_rad)  # Component perpendicular

    print(f"\nSteering vector decomposition:")
    print(f"  Parallel to prompted: {parallel_component:.4f} ({parallel_component/mag2*100:.1f}%)")
    print(f"  Orthogonal to prompted: {orthogonal_component:.4f} ({orthogonal_component/mag2*100:.1f}%)")

    # Create polar plot
    print("\nCreating polar plot...")
    fig_polar = create_polar_plot(angle_rad, mag1, mag2, cos_sim)
    polar_path = OUTPUT_DIR / "vector_geometry_polar.png"
    fig_polar.savefig(polar_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {polar_path}")

    # Create Cartesian plot
    print("Creating Cartesian plot...")
    fig_cart = create_cartesian_plot(angle_rad, mag1, mag2, cos_sim)
    cart_path = OUTPUT_DIR / "vector_geometry_cartesian.png"
    fig_cart.savefig(cart_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {cart_path}")

    print(f"\n{'='*80}")
    print("VISUALIZATION COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
