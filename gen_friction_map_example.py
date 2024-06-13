from frictionmap.friction_map_generator import generate_friction_map

if __name__ == '__main__':
    # Example  {'percentage_location_s': [0.0, 7.0], 'change': 'constant/linear', 'friction': [0.5]} <= from 0% to 7% of the track constant friction 0.5
    friction_descriptions = [{'percentage_location_s': [0.0, 68.0], 'change': 'constant', 'friction': [1.1]},
                            {'percentage_location_s': [68.0, 100.0], 'change': 'constant', 'friction': [0.5]},
                            ]

    # Any track from the f1tenth_racetracks folder
    track_name = "Austin"
    
    initial_mue = 1.0 # Initial friction value for the whole track
    cellwidth_m = 1.0 # Width of the grid cells of the friction map (cells are quadratic)
    inside_trackbound = 'left'  # if getting error: ValueError: No points given File "qhull.pyx", change this to right/left
    bool_show_plots = True # Show plots of the reference line, the friction map and the corresponding mue values

    generate_friction_map(track_name, initial_mue, cellwidth_m, friction_descriptions, inside_trackbound, bool_show_plots)