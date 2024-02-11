import matplotlib.pyplot as plt

# Values from the user's code
categories = ['Old Model', 'Updated Model', 'Updated Model\n6X Data', 'Updated Model\n80X Data']
values = [0.3609, 0.06038, 0.03725, 0.02399]
colors = ['red', 'orange', 'blue', 'green']

# Create bar plot
plt.figure(figsize=(10, 8))
bars = plt.bar(categories, values, color=colors)

# Add value labels above the bars in percentage format
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2%}', ha='center', va='bottom')  # ha='center' to align the text in the center of the bar

# Add titles and labels
plt.title('Model Performance Comparison')
plt.ylabel('Prediction Interval')
plt.figtext(0.45, 0.92, '(lower = better)', fontsize=10, color='red')  # "(lower = better)" in red

plt.grid(True, linestyle='--', which='major', color='grey', alpha=.25)

# Show the plot
plt.show()
