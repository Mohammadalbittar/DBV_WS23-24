def count_lanes(input_array):
    
    Lane1_ein, Lane2_ein, Lane3_ein, Lane4_ein, Lane1_aus, Lane2_aus, Lane3_aus, Lane4_aus = 0, 0, 0, 0, 0, 0, 0, 0
    for array in input_array:
        if array[1] == 1:
            Lane1_ein += 1
        if array[1] == 2:
            Lane2_ein += 1
        if array[1] == 3:
            Lane3_ein += 1
        if array[1] == 4:
            Lane4_ein += 1
        if array[2] == 1:
            Lane1_aus += 1
        if array[2] == 2:
            Lane2_aus += 1
        if array[2] == 3:
            Lane3_aus += 1
        if array[2] == 4:
            Lane4_aus += 1
        
    counted_lanes = [[Lane1_ein, Lane2_ein, Lane3_ein, Lane4_ein], [Lane1_aus, Lane2_aus, Lane3_aus, Lane4_aus]]
    print(counted_lanes)

    return counted_lanes





