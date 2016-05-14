# Simple cell tracking
from os.path import join as joinpath
import json
from math import sqrt


def get_dist(point1, point2):
    """Get Euclidean distance between two points, each a tuple/list of the form (x,y)"""
    return sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def get_all_dists(this_point, other_points):
    """Get list of distances between this_point (x,y) and list of other_points [(x,y), (x,y), ...]"""
    dists = []
    for other_point in other_points:
        dists.append(get_dist(this_point, other_point))
    return dists


def get_shortest_dist(this_point, other_points):
    """Get shortest (distance, index of other point) from this_point (x,y) and list of other_points [(x,y), (x,y), ...]"""
    dists = get_all_dists(this_point, other_points)
    min_dist, other_point_ind = min((val, idx) for (idx, val) in enumerate(dists))
    return min_dist, other_point_ind


def track_cells_basic(segmented_stats):
    """Build cell trajectories. Basically add a field to each cell inside segmented_stats that says what its previous
    label was. Cells in the first frame have dummy pre_label's that point to label 0."""

    # Make dummy previous components for 1st frame
    # These get updated at the end of each iteration
    prev_labels = [0]
    prev_centroids = [[-1, -1]]  # same order as prev_labels

    for frame in segmented_stats:

        for cell in frame['cells']:
            min_dist, prev_label_ind = get_shortest_dist(cell['centroid'], prev_centroids)
            cell['prev_label'] = prev_labels[prev_label_ind]

        prev_labels = []
        prev_centroids = []
        for cell in frame['cells']:
            prev_labels.append(cell['label'])
            prev_centroids.append(cell['centroid'])

    return segmented_stats


if __name__ == "__main__":
    output_dir = 'output'
    segmented_results_file = joinpath(output_dir, 'segmented_results.txt')

    with open(segmented_results_file) as f:
        segmented_results = json.load(f)

    print('Tracking cells')
    tracked_results = track_cells_basic(segmented_results)

    # Output results
    print('Outputting tracked results in JSON format')
    tracked_results_file = joinpath(output_dir, 'tracked_results.txt')
    with open(tracked_results_file, 'w') as f:
        json.dump(tracked_results, f, cls=json.JSONEncoder, indent=4, sort_keys=True)

    print('done.')
