from os.path import join as joinpath
import json
from math import sqrt


def distance(this, prev):
    """Get Euclidean distance between this position (x,y) and previous positions [(x,y), (x,y), ...]"""
    dists = []
    for pos in prev:
        dists.append(sqrt((this[0] - pos[0])**2 + (this[1] - pos[1])**2))
    return dists


def track_cells_basic(segmented_stats):
    """Build cell trajectories. Basically add a field to each cell inside segmented_stats that says what its previous
    label was."""

    # Make dummy previous components for 1st frame
    # These get updated at the end of each iteration
    prev_labels = [0]
    prev_centroids = [[-1, -1]]  # same order as prev_labels

    for frame in segmented_stats:

        for cell in frame['cells']:
            dists = distance(cell['centroid'], prev_centroids)
            min_dist, prev_label_ind = min((val, idx) for (idx, val) in enumerate(dists))
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
