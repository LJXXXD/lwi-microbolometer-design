import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Define a function to draw a rectangle with rounded corners
def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
    '''
    Draw a neural network cartoon using matplotilb.
    
    :param ax: matplotlib Axes object
    :param left: float, left coordinate
    :param right: float, right coordinate
    :param bottom: float, bottom coordinate
    :param top: float, top coordinate
    :param layer_sizes: list of int, size of each layer
    '''
    v_spacing = (top - bottom)/float(max(layer_sizes)//8)
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    # Nodes
    # for n, layer_size in enumerate(layer_sizes):
    #     nodes_to_draw = layer_size if layer_size <= 4 else max(layer_size // 8, 1)  # draw 1/8th of the nodes for large layers
    #     layer_top = v_spacing*(nodes_to_draw - 1)/2. + (top + bottom)/2.
    #     for m in range(nodes_to_draw):
    #         circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4., color='w', ec='k', zorder=4)
    #         ax.add_artist(circle)
    #         # Annotation for layer names
    #         if n == len(layer_sizes)-1:
    #             ax.text(n*h_spacing + left, layer_top - m*v_spacing, f'Output\n(softmax)', ha='center', va='center', zorder=5)
    #         elif n == 0:
    #             ax.text(n*h_spacing + left, layer_top - m*v_spacing, 'Input\nLayer', ha='center', va='center', zorder=5)
    #         else:
    #             ax.text(n*h_spacing + left, layer_top - m*v_spacing, f'FC-{layer_sizes[n]}\n(ReLU)', ha='center', va='center', zorder=5)
    #     # Add "..." for layers with more nodes
    #     if layer_size > 4:
    #         ax.text(n*h_spacing + left, layer_top - (nodes_to_draw * v_spacing), "...", ha='center', va='center', zorder=5)


    for n, layer_size in enumerate(layer_sizes):
        if layer_size <= 4:
            layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
            for m in range(layer_size):
                circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/2.5,
                                    color='w', ec='k', zorder=4)
                ax.add_artist(circle)
                # Annotation for layer names
                if n == len(layer_sizes)-1:
                    ax.text(n*h_spacing + left, layer_top - m*v_spacing, f'Output\n(softmax)', ha='center', va='center', zorder=5)
                elif n == 0:
                    ax.text(n*h_spacing + left, layer_top - m*v_spacing, 'Input\nLayer', ha='center', va='center', zorder=5)
                else:
                    ax.text(n*h_spacing + left, layer_top - m*v_spacing, f'FC-{layer_sizes[n]}\n(ReLU)', ha='center', va='center', zorder=5)
        else:
            layer_size = layer_size//8
            layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
            for m in range(layer_size):
                circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/2.5,
                                    color='w', ec='k', zorder=4)
                ax.add_artist(circle)
                # Annotation for layer names
                if n == len(layer_sizes)-1:
                    ax.text(n*h_spacing + left, layer_top - m*v_spacing, f'Output\n(softmax)', ha='center', va='center', zorder=5)
                elif n == 0:
                    ax.text(n*h_spacing + left, layer_top - m*v_spacing, 'Input\nLayer', ha='center', va='center', zorder=5)
                else:
                    ax.text(n*h_spacing + left, layer_top - m*v_spacing, f'FC-{layer_sizes[n]}\n(ReLU)', ha='center', va='center', zorder=5)

    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2. if layer_size_a <= 4 else v_spacing*(layer_size_a//8 - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2. if layer_size_b <= 4 else v_spacing*(layer_size_b//8 - 1)/2. + (top + bottom)/2.
        m = 1
        if layer_size_b <=4:
            for o in range(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                    [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k')
                ax.add_artist(line)
        else:
            for o in range(layer_size_b//8):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                    [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k')
                ax.add_artist(line)

# Create the figure
fig = plt.figure(figsize=(12, 12))
ax = fig.gca()
ax.axis('off')

# Layer sizes of the MLP (input layer size and output layer size are hypothetical)
layer_sizes = [3, 32, 32, 64, 4]

# Draw the neural network
draw_neural_net(ax, .1, .9, .1, .9, layer_sizes)

# Additional annotations for batch normalization and dropout
# ax.text(0.5, 0.93, 'Batch Normalization', ha='center', va='center', fontsize=12)
# ax.text(0.5, 0.06, '20% Dropout', ha='center', va='center', fontsize=12)

# Add rectangles for layers
# for i in range(1, 4):
#     # Add rectangle for batch normalization
#     bbox_props = dict(boxstyle="round,pad=0.3", fc="cyan", ec="b", lw=2)
#     ax.text(i*0.2, 0.9, f'BatchNorm1d({layer_sizes[i]})', ha='center', va='center', fontsize=10, bbox=bbox_props)

# Add rectangle for dropout layer
# bbox_props = dict(boxstyle="round,pad=0.3", fc="skyblue", ec="b", lw=2)
# ax.text(0.5, 0.09, 'Dropout(p=0.2)', ha='center', va='center', fontsize=10, bbox=bbox_props)

plt.show()
