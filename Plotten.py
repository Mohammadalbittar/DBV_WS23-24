import matplotlib.pyplot as plt

def count_lanes(input_array):
    Lane1_ein, Lane2_ein, Lane3_ein, Lane4_ein, Lane1_aus, Lane2_aus, Lane3_aus, Lane4_aus = 0, 0, 0, 0, 0, 0, 0, 0
    for array in input_array:
        if array[0] == 1:
            Lane1_ein += 1
        if array[0] == 2:
            Lane2_ein += 1
        if array[0] == 3:
            Lane3_ein += 1
        if array[0] == 4:
            Lane4_ein += 1
        if array[1] == 1:
            Lane1_aus += 1
        if array[1] == 2:
            Lane2_aus += 1
        if array[1] == 3:
            Lane3_aus += 1
        if array[1] == 4:
            Lane4_aus += 1
        
    counted_lanes = [[Lane1_ein, Lane2_ein, Lane3_ein, Lane4_ein], [Lane1_aus, Lane2_aus, Lane3_aus, Lane4_aus]]
    print('counted Lanes: ', counted_lanes)

    labels = ['Richtung 1', 'Richtung 2', 'Richtung 3', 'Richtung 4']
    x = range(len(labels))

    fig, ax = plt.subplots()
    width = 0.35

    rects1 = ax.bar(x, counted_lanes[0], width, label='Einfahrt')
    rects2 = ax.bar([i + width for i in x], counted_lanes[1], width, label='Ausfahrt')

    ax.set_ylabel('Anzahl Fahrzeuge')
    ax.set_title('Anzahl detektierte Fahrzeuge pro Richtung')
    ax.set_xticks([i + width / 2 for i in x])
    ax.set_xticklabels(labels)
    ax.legend()

    plt.show()


